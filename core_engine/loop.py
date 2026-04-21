from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, TypedDict, cast

import numpy as np
import pandas as pd

from attacker_engine.agent import AttackerAgent
from attacker_engine.strategies import normalize_strategy_mix
from defender_engine.agent import DefenderAgent
from evaluation import compute_detection_metrics, compute_system_cost_metrics, simulate_economic_rewards
from features import extract_node_features
from modeling import benchmark_fraud_models, train_fraud_model
from simulation import build_network_simulation
from utils.constants import FEATURE_COLUMNS


class SimulationParams(TypedDict):
    honest_anomaly_rate: float
    noise_level: float
    delayed_observation_rate: float
    temporal_drift_rate: float
    smart_strategy_mix: Dict[str, float]


MIMICRY_BASE_WEIGHT = 0.14
MIMICRY_BONUS_WEIGHT = 0.75
DELAY_BASE_WEIGHT = 0.08
DELAY_BONUS_WEIGHT = 0.50
NOISE_BASE_RATE = 0.10
NOISE_BASE_LEVEL = 0.18
DELAY_BASE_OBSERVATION_RATE = 0.06
BASE_TEMPORAL_DRIFT_RATE = 0.05
SELECTED_STRATEGY_BOOST = 0.25
SELECTED_STRATEGY_ECONOMIC_BONUS = 0.25
SELECTED_STRATEGY_DETECTION_BONUS = 0.15
SELECTED_STRATEGY_FRAUD_PROFIT_BONUS = 0.08
STRATEGY_NAMES = (
    "low_and_slow",
    "burst_attack",
    "camouflage",
    "perfect_mimic",
    "slow_drift",
    "decoy_attacker",
    "mixed_cluster",
)


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
    selected_attacker_strategy: str


@dataclass(frozen=True)
class ClosedLoopArtifacts:
    iterations: List[ClosedLoopIteration]
    final_model: object
    benchmark: pd.DataFrame
    last_features: pd.DataFrame
    simulation_params: Dict[str, float]
    evolution_history: Dict[str, List[float] | List[Dict[str, float]]]


@dataclass(frozen=True)
class StepObservation:
    iteration: int
    difficulty_level: str
    strategy_mix: Dict[str, float]
    defender_threshold: float
    simulation_params: SimulationParams


@dataclass(frozen=True)
class StepOutcome:
    result: ClosedLoopIteration
    next_observation: StepObservation | None


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
        "temporal_drift_rate": float(np.clip(BASE_TEMPORAL_DRIFT_RATE + delayed / scale, 0.02, 0.50)),
    }


def _normalize_sim_params(sim_params: SimulationParams, strategy_mix: Mapping[str, float]) -> SimulationParams:
    out = dict(sim_params)
    out["smart_strategy_mix"] = dict(normalize_strategy_mix(dict(strategy_mix)))
    out["honest_anomaly_rate"] = float(np.clip(out.get("honest_anomaly_rate", NOISE_BASE_RATE), 0.01, 0.60))
    out["noise_level"] = float(np.clip(out.get("noise_level", NOISE_BASE_LEVEL), 0.01, 0.95))
    out["delayed_observation_rate"] = float(np.clip(out.get("delayed_observation_rate", DELAY_BASE_OBSERVATION_RATE), 0.01, 0.60))
    out["temporal_drift_rate"] = float(np.clip(out.get("temporal_drift_rate", BASE_TEMPORAL_DRIFT_RATE), 0.01, 0.60))
    return cast(SimulationParams, out)


def _strategy_feedback(
    selected_strategy: str,
    failures: Dict[str, int],
    system_costs: Dict[str, float],
    economic_summary: Dict[str, float],
    tp: int,
    fn: int,
) -> Dict[str, Dict[str, float]]:
    leakage = float(economic_summary.get("pct_reward_lost_to_fraud", 0.0)) / 100.0
    fraud_profit = float(economic_summary.get("total_fraud_profit", 0.0))
    defender_cost = float(system_costs.get("total_system_cost", 0.0))
    attacker_detection_pressure = float(tp / max(tp + fn, 1))
    mimicry = float(failures.get("mimicry_failure", 0))
    delayed = float(failures.get("delayed_detection", 0))
    noise = float(failures.get("noise_failure", 0))

    base_gain = {
        "low_and_slow": 0.35 * leakage + 0.10 * fraud_profit,
        "burst_attack": 0.45 * leakage + 0.16 * fraud_profit,
        "camouflage": 0.30 * leakage + 0.12 * mimicry,
        "perfect_mimic": 0.32 * leakage + 0.16 * mimicry,
        "slow_drift": 0.26 * leakage + 0.14 * delayed,
        "decoy_attacker": 0.18 * leakage + 0.10 * noise,
        "mixed_cluster": 0.33 * leakage + 0.11 * delayed,
    }
    base_penalty = {
        "low_and_slow": 0.45 * attacker_detection_pressure + 0.01 * defender_cost,
        "burst_attack": 0.65 * attacker_detection_pressure + 0.01 * defender_cost,
        "camouflage": 0.40 * attacker_detection_pressure + 0.008 * defender_cost,
        "perfect_mimic": 0.36 * attacker_detection_pressure + 0.008 * defender_cost,
        "slow_drift": 0.30 * attacker_detection_pressure + 0.006 * defender_cost,
        "decoy_attacker": 0.50 * attacker_detection_pressure + 0.007 * defender_cost,
        "mixed_cluster": 0.42 * attacker_detection_pressure + 0.008 * defender_cost,
    }
    feedback = {
        strategy: {"economic_gain": float(base_gain[strategy]), "detection_penalty": float(base_penalty[strategy])}
        for strategy in base_gain
    }
    if selected_strategy in feedback:
        feedback[selected_strategy]["economic_gain"] += (
            SELECTED_STRATEGY_ECONOMIC_BONUS * leakage
            + SELECTED_STRATEGY_FRAUD_PROFIT_BONUS * fraud_profit
        )
        feedback[selected_strategy]["detection_penalty"] += SELECTED_STRATEGY_DETECTION_BONUS * attacker_detection_pressure
    return feedback


class AdversarialFraudEnvironment:
    def __init__(self, iterations: int, seed: int, total_nodes: int, time_steps: int) -> None:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.total_nodes = int(total_nodes)
        self.time_steps = int(time_steps)
        self.rng = np.random.default_rng(seed)
        self.train_df = pd.DataFrame()
        self.final_model = None
        self.sim_params: SimulationParams = {
            "honest_anomaly_rate": 0.12,
            "noise_level": 0.30,
            "delayed_observation_rate": 0.06,
            "temporal_drift_rate": BASE_TEMPORAL_DRIFT_RATE,
            "smart_strategy_mix": {
                strategy: 1.0 / len(STRATEGY_NAMES)
                for strategy in STRATEGY_NAMES
            },
        }
        self.history: Dict[str, List[float] | List[Dict[str, float]]] = {
            "attacker_strategy_distribution": [],
            "defender_threshold": [],
            "system_cost": [],
            "fraud_leakage": [],
        }

    def _observation(self, iteration: int, attacker: AttackerAgent, defender: DefenderAgent) -> StepObservation:
        return StepObservation(
            iteration=iteration,
            difficulty_level=_curriculum_difficulty(iteration, self.iterations),
            strategy_mix=dict(attacker.strategy_mix),
            defender_threshold=float(defender.threshold),
            simulation_params=_normalize_sim_params(self.sim_params, attacker.strategy_mix),
        )

    def step(self, iteration: int, attacker_agent: AttackerAgent, defender_agent: DefenderAgent) -> StepOutcome:
        observation = self._observation(iteration=iteration, attacker=attacker_agent, defender=defender_agent)
        chosen_strategy = attacker_agent.sample_strategy(self.rng)
        acted_mix = dict(observation.strategy_mix)
        acted_mix[chosen_strategy] = float(acted_mix.get(chosen_strategy, 0.0) + SELECTED_STRATEGY_BOOST)
        acted_mix = normalize_strategy_mix(acted_mix)
        params = _normalize_sim_params(observation.simulation_params, strategy_mix=acted_mix)

        sim = build_network_simulation(
            total_nodes=self.total_nodes,
            time_steps=self.time_steps,
            seed=self.seed + iteration,
            difficulty_level=observation.difficulty_level,
            model_for_adaptation=self.final_model.model if self.final_model is not None else None,
            fraud_score_threshold=defender_agent.threshold,
            feature_columns=FEATURE_COLUMNS if self.final_model is not None else (),
            smart_strategy_mix=params.get("smart_strategy_mix"),
            honest_anomaly_rate=float(params["honest_anomaly_rate"]),
            noise_level=float(params["noise_level"]),
            delayed_observation_rate=float(params["delayed_observation_rate"]),
            adaptation_growth=float(params["temporal_drift_rate"]),
        )
        df = extract_node_features(sim)

        if self.final_model is not None:
            pre_probs = self.final_model.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
            hard_cases = _hard_negative_mining(df, probs=pre_probs, threshold=defender_agent.threshold)
        else:
            hard_cases = df.iloc[0:0].copy()
        self.train_df = pd.concat([self.train_df, df, hard_cases], ignore_index=True)

        self.final_model = train_fraud_model(self.train_df, model_path="model.pkl", random_state=self.seed + iteration)
        probs = self.final_model.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        fraud_scores = {int(node_id): float(score) for node_id, score in zip(df["node_id"], probs)}
        selected_nodes = defender_agent.select_nodes(fraud_scores=fraud_scores)

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

        failures = _failure_breakdown(df=df, probs=probs, threshold=defender_agent.threshold)
        node_labels = {int(node_id): str(label_name) for node_id, label_name in zip(df["node_id"], df["label_name"])}
        node_regions = {int(node_id): str(region) for node_id, region in zip(df["node_id"], df["region"])}
        detection_metrics = compute_detection_metrics(
            fraud_scores_over_time=[fraud_scores],
            node_labels=node_labels,
            threshold=defender_agent.threshold,
            node_regions=node_regions,
            high_value_regions={"new_york", "tokyo", "london", "singapore"},
        )
        system_costs = compute_system_cost_metrics(
            per_node=detection_metrics.per_node,
            false_positives_over_time=detection_metrics.false_positives_over_time,
        )
        economic = simulate_economic_rewards(
            fraud_scores_over_time=[fraud_scores],
            node_labels=node_labels,
            threshold=defender_agent.threshold,
            node_regions=node_regions,
        )
        economic_summary = economic.summary

        strategy_feedback = _strategy_feedback(
            selected_strategy=chosen_strategy,
            failures=failures,
            system_costs=system_costs,
            economic_summary=economic_summary,
            tp=tp,
            fn=fn,
        )
        strategy_mix = attacker_agent.update(strategy_feedback=strategy_feedback)
        defender_agent.update_policy(
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            delayed_detection_rate=float(detection_metrics.summary.get("avg_detection_delay", 0.0)),
            system_cost=float(system_costs.get("total_system_cost", 0.0)),
        )

        failure_driven = _compute_augmentation_params(failures)
        self.sim_params = _normalize_sim_params(failure_driven, strategy_mix=strategy_mix)

        round_metrics = dict(self.final_model.metrics)
        round_metrics.update(
            {
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
                "avg_detection_delay": float(detection_metrics.summary.get("avg_detection_delay", 0.0)),
                "avg_false_positives_over_time": float(detection_metrics.summary.get("avg_false_positives_over_time", 0.0)),
                "total_system_cost": float(system_costs.get("total_system_cost", 0.0)),
                "fraud_leakage_pct": float(economic_summary.get("pct_reward_lost_to_fraud", 0.0)),
                "total_fraud_profit": float(economic_summary.get("total_fraud_profit", 0.0)),
            }
        )

        self.history["attacker_strategy_distribution"].append(dict(strategy_mix))
        self.history["defender_threshold"].append(float(defender_agent.threshold))
        self.history["system_cost"].append(float(system_costs.get("total_system_cost", 0.0)))
        self.history["fraud_leakage"].append(float(economic_summary.get("pct_reward_lost_to_fraud", 0.0)))

        iteration_result = ClosedLoopIteration(
            iteration=iteration,
            difficulty_level=observation.difficulty_level,
            hard_negative_count=int(len(hard_cases)),
            metrics=round_metrics,
            selected_nodes=selected_nodes,
            failure_breakdown=failures,
            attacker_strategy_mix=dict(strategy_mix),
            defender_threshold=float(defender_agent.threshold),
            selected_attacker_strategy=chosen_strategy,
        )
        next_observation = (
            self._observation(iteration=iteration + 1, attacker=attacker_agent, defender=defender_agent)
            if (iteration + 1) < self.iterations
            else None
        )
        return StepOutcome(result=iteration_result, next_observation=next_observation)


def run_closed_loop(iterations: int = 3, seed: int = 42, total_nodes: int = 220, time_steps: int = 18) -> ClosedLoopArtifacts:
    environment = AdversarialFraudEnvironment(
        iterations=iterations,
        seed=seed,
        total_nodes=total_nodes,
        time_steps=time_steps,
    )
    attacker = AttackerAgent(temperature=0.25, epsilon=0.08, use_ucb_selection=True)
    defender = DefenderAgent(threshold=0.5, budget_ratio=0.07, min_budget=3)

    rounds: List[ClosedLoopIteration] = []
    for i in range(iterations):
        outcome = environment.step(iteration=i, attacker_agent=attacker, defender_agent=defender)
        rounds.append(outcome.result)

    benchmark = benchmark_fraud_models(train_dataset=environment.train_df, test_dataset=environment.train_df, random_state=seed)
    return ClosedLoopArtifacts(
        iterations=rounds,
        final_model=environment.final_model,
        benchmark=benchmark,
        last_features=environment.train_df,
        simulation_params=dict(environment.sim_params),
        evolution_history=dict(environment.history),
    )
