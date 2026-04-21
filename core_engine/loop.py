from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TypedDict

import numpy as np
import pandas as pd

from attacker_engine.agent import AttackerAgent
from utils.constants import FEATURE_COLUMNS
from defender_engine.agent import DefenderAgent
from evaluation import compute_detection_metrics, compute_system_cost_metrics
from features import extract_node_features
from modeling import benchmark_fraud_models, train_fraud_model
from simulation import build_network_simulation


class SimulationParams(TypedDict):
    honest_anomaly_rate: float
    noise_level: float
    delayed_observation_rate: float
    smart_strategy_mix: Dict[str, float]

MIMICRY_BASE_WEIGHT = 0.14
MIMICRY_BONUS_WEIGHT = 0.75
DELAY_BASE_WEIGHT = 0.08
DELAY_BONUS_WEIGHT = 0.50
NOISE_BASE_RATE = 0.10
NOISE_BASE_LEVEL = 0.18
DELAY_BASE_OBSERVATION_RATE = 0.06


@dataclass(frozen=True)
class ClosedLoopIteration:
    iteration: int
    difficulty_level: str
    hard_negative_count: int
    metrics: Dict[str, float]
    selected_nodes: List[int]
    failure_breakdown: Dict[str, int]
    attacker_strategy_mix: Dict[str, float]
    defender_threshold: float


@dataclass(frozen=True)
class ClosedLoopArtifacts:
    iterations: List[ClosedLoopIteration]
    final_model: object
    benchmark: pd.DataFrame
    last_features: pd.DataFrame
    simulation_params: Dict[str, float]


def _curriculum_difficulty(round_idx: int, total_rounds: int) -> str:
    if total_rounds <= 1:
        return "medium"
    progress = float(round_idx / max(total_rounds - 1, 1))
    if progress < 0.34:
        return "easy"
    if progress < 0.67:
        return "medium"
    if progress < 0.9:
        return "hard"
    return "extreme"


def _failure_breakdown(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> Dict[str, int]:
    out = {"mimicry_failure": 0, "noise_failure": 0, "delayed_detection": 0, "other_failure": 0}
    if df.empty:
        return out
    y = df["label"].astype(int).to_numpy(dtype=int)
    pred = (probs >= threshold).astype(int)
    mismatch_vals = pd.to_numeric(df.get("claimed_inferred_distance_mismatch_mean", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    inconsistency_vals = pd.to_numeric(df.get("latency_inconsistency_score_mean", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for idx in range(len(df)):
        if int(y[idx]) == int(pred[idx]):
            continue
        mismatch = float(mismatch_vals[idx])
        inconsistency = float(inconsistency_vals[idx])
        if int(y[idx]) == 1 and mismatch < 350 and inconsistency < 0.30:
            out["mimicry_failure"] += 1
        elif int(y[idx]) == 0 and inconsistency > 0.75:
            out["noise_failure"] += 1
        elif int(y[idx]) == 1 and float(probs[idx]) >= (threshold * 0.85):
            out["delayed_detection"] += 1
        else:
            out["other_failure"] += 1
    return out


def _hard_negative_mining(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    y = df["label"].astype(int).to_numpy()
    pred = (probs >= threshold).astype(int)
    false_neg = (y == 1) & (pred == 0)
    false_pos = (y == 0) & (pred == 1)
    hard_negatives = df.loc[false_neg | false_pos].copy()
    if hard_negatives.empty:
        return hard_negatives
    return pd.concat([hard_negatives] * 2, ignore_index=True)


def _compute_augmentation_params(failure_breakdown: Dict[str, int]) -> SimulationParams:
    mimic = float(failure_breakdown.get("mimicry_failure", 0))
    noise = float(failure_breakdown.get("noise_failure", 0))
    delayed = float(failure_breakdown.get("delayed_detection", 0))
    scale = max(mimic + noise + delayed, 1.0)
    return {
        "smart_strategy_mix": {
            "camouflage": float(np.clip(MIMICRY_BASE_WEIGHT + mimic / scale, 0.05, 0.78)),
            "perfect_mimic": float(np.clip(0.10 + MIMICRY_BONUS_WEIGHT * mimic / scale, 0.05, 0.72)),
            "slow_drift": float(np.clip(DELAY_BASE_WEIGHT + delayed / scale, 0.05, 0.68)),
            "mixed_cluster": float(np.clip(DELAY_BASE_WEIGHT + DELAY_BONUS_WEIGHT * delayed / scale, 0.04, 0.55)),
            "low_and_slow": 0.08,
            "burst_attack": 0.08,
            "decoy_attacker": 0.12,
        },
        "honest_anomaly_rate": float(np.clip(NOISE_BASE_RATE + noise / scale, 0.06, 0.42)),
        "noise_level": float(np.clip(NOISE_BASE_LEVEL + noise / scale, 0.10, 0.78)),
        "delayed_observation_rate": float(np.clip(DELAY_BASE_OBSERVATION_RATE + delayed / scale, 0.02, 0.45)),
    }


def run_closed_loop(iterations: int = 3, seed: int = 42, total_nodes: int = 220, time_steps: int = 18) -> ClosedLoopArtifacts:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    attacker = AttackerAgent()
    defender = DefenderAgent(threshold=0.5, budget=max(3, int(total_nodes * 0.07)))
    sim_params: SimulationParams = {
        "honest_anomaly_rate": 0.12,
        "noise_level": 0.30,
        "delayed_observation_rate": 0.06,
        "smart_strategy_mix": dict(attacker.strategy_mix),
    }

    rounds: List[ClosedLoopIteration] = []
    final_model = None
    train_df = pd.DataFrame()

    for i in range(iterations):
        difficulty = _curriculum_difficulty(i, iterations)
        sim = build_network_simulation(
            total_nodes=total_nodes,
            time_steps=time_steps,
            seed=seed + i,
            difficulty_level=difficulty,
            model_for_adaptation=final_model.model if final_model is not None else None,
            fraud_score_threshold=defender.threshold,
            feature_columns=FEATURE_COLUMNS if final_model is not None else (),
            smart_strategy_mix=sim_params.get("smart_strategy_mix"),
            honest_anomaly_rate=float(sim_params["honest_anomaly_rate"]),
            noise_level=float(sim_params["noise_level"]),
            delayed_observation_rate=float(sim_params["delayed_observation_rate"]),
        )
        df = extract_node_features(sim)

        if final_model is not None:
            pre_probs = final_model.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
            hard_cases = _hard_negative_mining(df, probs=pre_probs, threshold=defender.threshold)
        else:
            hard_cases = df.iloc[0:0].copy()
        train_df = pd.concat([train_df, df, hard_cases], ignore_index=True)

        final_model = train_fraud_model(train_df, model_path="model.pkl", random_state=seed + i)
        probs = final_model.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        selected_nodes = defender.select_nodes({int(nid): float(score) for nid, score in zip(df["node_id"], probs)})

        y_true = df["label"].astype(int).to_numpy(dtype=int)
        y_pred = np.zeros_like(y_true)
        node_to_idx = {int(node_id): idx for idx, node_id in enumerate(df["node_id"].tolist())}
        for node_id in selected_nodes:
            idx = node_to_idx.get(int(node_id))
            if idx is not None:
                y_pred[idx] = 1
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp_rate = float(fp / max(fp + tn, 1))
        fn_rate = float(fn / max(fn + tp, 1))

        failures = _failure_breakdown(df=df, probs=probs, threshold=defender.threshold)
        fraud_scores = {int(node_id): float(score) for node_id, score in zip(df["node_id"], probs)}
        node_labels = {int(node_id): str(label_name) for node_id, label_name in zip(df["node_id"], df["label_name"])}
        detection_metrics = compute_detection_metrics(
            fraud_scores_over_time=[fraud_scores],
            node_labels=node_labels,
            threshold=defender.threshold,
        )
        system_costs = compute_system_cost_metrics(
            per_node=detection_metrics.per_node,
            false_positives_over_time=detection_metrics.false_positives_over_time,
        )

        rewards = {
            "camouflage": float(failures.get("mimicry_failure", 0)),
            "perfect_mimic": float(failures.get("mimicry_failure", 0)),
            "slow_drift": float(failures.get("delayed_detection", 0)),
            "mixed_cluster": float(failures.get("delayed_detection", 0) * 0.5),
            "low_and_slow": float(max(int((y_true == 1).sum()) - fn, 0)),
            "burst_attack": float(max(int((y_true == 1).sum()) - fn - tp * 0.25, 0)),
            "decoy_attacker": float(failures.get("noise_failure", 0) * 0.5),
        }
        strategy_mix = attacker.update(rewards)
        defender.update_policy(false_positive_rate=fp_rate, false_negative_rate=fn_rate)
        sim_params = _compute_augmentation_params(failures)
        sim_params["smart_strategy_mix"] = dict(strategy_mix)

        round_metrics = dict(final_model.metrics)
        round_metrics.update(
            {
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
                "avg_detection_delay": float(detection_metrics.summary.get("avg_detection_delay", 0.0)),
                "avg_false_positives_over_time": float(detection_metrics.summary.get("avg_false_positives_over_time", 0.0)),
                "total_system_cost": float(system_costs.get("total_system_cost", 0.0)),
            }
        )
        rounds.append(
            ClosedLoopIteration(
                iteration=i,
                difficulty_level=difficulty,
                hard_negative_count=int(len(hard_cases)),
                metrics=round_metrics,
                selected_nodes=selected_nodes,
                failure_breakdown=failures,
                attacker_strategy_mix=dict(strategy_mix),
                defender_threshold=float(defender.threshold),
            )
        )

    benchmark = benchmark_fraud_models(train_dataset=train_df, test_dataset=train_df, random_state=seed)
    return ClosedLoopArtifacts(
        iterations=rounds,
        final_model=final_model,
        benchmark=benchmark,
        last_features=train_df,
        simulation_params=sim_params,
    )
