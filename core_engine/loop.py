from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, TypedDict, cast

import numpy as np
import pandas as pd

from attacker_engine.agent import AttackerAgent
from attacker_engine.strategies import normalize_strategy_mix
from defender_engine.agent import DefenderAgent
from evaluation import compute_detection_metrics, compute_system_cost_metrics, simulate_economic_rewards
from evaluation.visualization import (
    plot_cost_vs_time,
    plot_defender_threshold_over_time,
    plot_equilibrium_detection,
    plot_fraud_leakage_vs_time,
    plot_strategy_distribution_over_time,
)
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
LEGACY_STRATEGIC_MEMORY_PATH = Path("strategic_memory.json")
MEMORY_DIR = Path("memory")
ATTACKER_MEMORY_PATH = MEMORY_DIR / "attacker_memory.json"
DEFENDER_MEMORY_PATH = MEMORY_DIR / "defender_memory.json"
ADVERSARIAL_SCENARIOS = (
    "steady_state_low_level_fraud",
    "coordinated_attack_campaign",
    "sudden_attack_surge",
    "defender_overload",
    "stealth_long_term_infiltration",
)
# Maps budget ratio to normalized defender pressure used by attacker forecasts.
DEFENDER_INTENSITY_SCALE = 8.0
# Converts recent leakage (%) into a bounded profit signal for attacker planning.
FRAUD_PROFIT_FROM_LEAKAGE_SCALE = 0.12
FRAUD_PROFIT_CAP = 40.0
# Inner best-response loop: stop when max policy delta falls below this epsilon.
INNER_CONVERGENCE_EPSILON = 0.01
# Cross-run memory decay: multiply historical reward estimates by this rate per run.
MEMORY_DECAY_RATE = 0.90
# Dirichlet concentration multiplier: scales observed mix into alpha for Thompson sampling.
DIRICHLET_CONCENTRATION = 12.0
# Consecutive strategy repetition: penalty scales with repeat count, capped below.
STRATEGY_REPEAT_PENALTY_SCALE = 0.08
STRATEGY_REPEAT_PENALTY_CAP = 0.30
# Attacker reward hardening: when leakage fraction exceeds this, gains are reduced.
LEAKAGE_HARDENING_THRESHOLD = 0.15
LEAKAGE_HARDENING_SCALE = 0.50
# Equilibrium type thresholds using cost coefficient of variation (CV = std/mean).
EQUILIBRIUM_CV_STABLE_THRESHOLD = 0.10
EQUILIBRIUM_CV_CHAOTIC_THRESHOLD = 0.40
# Success-rate bias scale for warm-starting from historical memory outcomes.
MEMORY_SUCCESS_RATE_BIAS_SCALE = 0.05
# Scenario-specific pressure injected into strategy mix and environment controls.
SCENARIO_PRESSURE = {
    "steady_state_low_level_fraud": {
        "low_and_slow": 0.10,
        "slow_drift": 0.08,
        "noise_level": -0.04,
    },
    "coordinated_attack_campaign": {"mixed_cluster": 0.16, "camouflage": 0.08},
    "sudden_attack_surge": {"burst_attack": 0.20, "noise_level": 0.08},
    "defender_overload": {"decoy_attacker": 0.18, "delayed_observation_rate": 0.09},
    "stealth_long_term_infiltration": {"low_and_slow": 0.12, "slow_drift": 0.14, "temporal_drift_rate": 0.05},
}


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
    defender_budget_ratio: float
    defender_selection_strategy: str
    selected_attacker_strategy: str
    selected_adversarial_scenario: str
    equilibrium_detected: bool


@dataclass(frozen=True)
class ClosedLoopArtifacts:
    iterations: List[ClosedLoopIteration]
    final_model: object
    benchmark: pd.DataFrame
    last_features: pd.DataFrame
    simulation_params: Dict[str, float]
    evolution_history: Dict[str, List[float] | List[Dict[str, float]]]
    system_analysis: Dict[str, object]


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


def _load_json_memory(path: Path, default_payload: Mapping[str, object]) -> Dict[str, object]:
    if not path.exists():
        return dict(default_payload)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(default_payload)
    if not isinstance(payload, dict):
        return dict(default_payload)
    return payload


def _save_json_memory(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def detect_equilibrium(
    history: Mapping[str, List[float] | List[Dict[str, float]]],
    consecutive_steps: int = 3,
    strategy_tolerance: float = 0.03,
    threshold_tolerance: float = 0.02,
    budget_tolerance: float = 0.02,
    selection_tolerance: float = 0.20,
    cost_tolerance_ratio: float = 0.03,
) -> Dict[str, object]:
    """Detect convergence using recent attacker-mix, threshold, and cost stability.

    The tolerances are small relative-change cutoffs for short-horizon closed-loop runs.
    """
    strategy_history_raw = history.get("attacker_strategy_distribution", [])
    threshold_history_raw = history.get("defender_threshold", [])
    budget_history_raw = history.get("defender_budget_ratio", [])
    selection_history_raw = history.get("defender_selection_strategy_id", [])
    cost_history_raw = history.get("system_cost", [])
    strategy_history = list(strategy_history_raw) if isinstance(strategy_history_raw, list) else []
    threshold_history = list(threshold_history_raw) if isinstance(threshold_history_raw, list) else []
    budget_history = list(budget_history_raw) if isinstance(budget_history_raw, list) else []
    selection_history = list(selection_history_raw) if isinstance(selection_history_raw, list) else []
    cost_history = list(cost_history_raw) if isinstance(cost_history_raw, list) else []
    window_size = max(int(consecutive_steps), 2)
    window = window_size  # alias used throughout _window_stats closure
    min_len = min(len(strategy_history), len(threshold_history), len(budget_history), len(selection_history), len(cost_history))

    def _window_stats(end_idx: int) -> Dict[str, float | bool]:
        start = max(end_idx - window + 1, 0)
        recent_strategies = strategy_history[start : end_idx + 1]
        names = sorted({k for snapshot in recent_strategies if isinstance(snapshot, dict) for k in snapshot.keys()})
        deltas: List[float] = []
        for idx in range(1, len(recent_strategies)):
            prev = recent_strategies[idx - 1] if isinstance(recent_strategies[idx - 1], dict) else {}
            curr = recent_strategies[idx] if isinstance(recent_strategies[idx], dict) else {}
            delta = float(sum(abs(float(curr.get(name, 0.0)) - float(prev.get(name, 0.0))) for name in names))
            deltas.append(delta)
        strategy_delta = float(max(deltas)) if deltas else float("inf")
        recent_threshold = np.array([float(v) for v in threshold_history[start : end_idx + 1]], dtype=float)
        threshold_deltas = np.abs(np.diff(recent_threshold))
        threshold_delta = float(np.max(threshold_deltas)) if threshold_deltas.size > 0 else float("inf")
        recent_budget = np.array([float(v) for v in budget_history[start : end_idx + 1]], dtype=float)
        budget_deltas = np.abs(np.diff(recent_budget))
        budget_delta = float(np.max(budget_deltas)) if budget_deltas.size > 0 else float("inf")
        recent_selection = np.array([float(v) for v in selection_history[start : end_idx + 1]], dtype=float)
        selection_deltas = np.abs(np.diff(recent_selection))
        selection_delta = float(np.max(selection_deltas)) if selection_deltas.size > 0 else float("inf")
        recent_cost = np.array([float(v) for v in cost_history[start : end_idx + 1]], dtype=float)
        cost_deltas = np.abs(np.diff(recent_cost))
        if cost_deltas.size > 0:
            largest_cost_delta = float(np.max(cost_deltas))
            baseline_cost = float(np.mean(np.abs(recent_cost)))
            cost_delta_ratio = float(largest_cost_delta / max(baseline_cost, 1.0))
        else:
            cost_delta_ratio = float("inf")
        is_eq = bool(
            strategy_delta <= strategy_tolerance
            and threshold_delta <= threshold_tolerance
            and budget_delta <= budget_tolerance
            and selection_delta <= selection_tolerance
            and cost_delta_ratio <= cost_tolerance_ratio
        )
        return {
            "is_equilibrium": is_eq,
            "strategy_delta": strategy_delta,
            "threshold_delta": threshold_delta,
            "budget_delta": budget_delta,
            "selection_delta": selection_delta,
            "cost_delta_ratio": cost_delta_ratio,
        }

    if min_len < window:
        return {
            "is_equilibrium": False,
            "equilibrium_reached": False,
            "time_to_equilibrium": -1,
            "equilibrium_state": {},
            "strategy_delta": float("inf"),
            "threshold_delta": float("inf"),
            "budget_delta": float("inf"),
            "selection_delta": float("inf"),
            "cost_delta_ratio": float("inf"),
        }

    first_eq_idx = -1
    for idx in range(window - 1, min_len):
        stats = _window_stats(idx)
        if bool(stats["is_equilibrium"]):
            first_eq_idx = idx
            break
    latest_stats = _window_stats(min_len - 1)
    latest_policy = {
        "threshold": float(threshold_history[min_len - 1]),
        "budget_ratio": float(budget_history[min_len - 1]),
        "selection_strategy_id": float(selection_history[min_len - 1]),
    }
    latest_state = {
        "attacker_strategy_distribution": strategy_history[min_len - 1] if isinstance(strategy_history[min_len - 1], dict) else {},
        "defender_policy": latest_policy,
        "system_cost": float(cost_history[min_len - 1]),
    }
    equilibrium_reached = bool(latest_stats["is_equilibrium"])
    return {
        "is_equilibrium": equilibrium_reached,
        "equilibrium_reached": equilibrium_reached,
        "time_to_equilibrium": int(first_eq_idx),
        "equilibrium_state": latest_state if equilibrium_reached else {},
        "strategy_delta": float(latest_stats["strategy_delta"]),
        "threshold_delta": float(latest_stats["threshold_delta"]),
        "budget_delta": float(latest_stats["budget_delta"]),
        "selection_delta": float(latest_stats["selection_delta"]),
        "cost_delta_ratio": float(latest_stats["cost_delta_ratio"]),
    }


def analyze_system_dynamics(
    history: Mapping[str, List[float] | List[Dict[str, float]]],
    rounds: List[ClosedLoopIteration],
) -> Dict[str, object]:
    latest_eq = detect_equilibrium(history)
    strategy_history = history.get("attacker_strategy_distribution", [])
    final_strategy_mix = strategy_history[-1] if isinstance(strategy_history, list) and strategy_history else {}
    dominant = "none"
    dominance = 0.0
    dominant_strategies: Dict[str, float] = {}
    if isinstance(final_strategy_mix, dict) and final_strategy_mix:
        sorted_strategies = sorted(final_strategy_mix.items(), key=lambda x: x[1], reverse=True)
        dominant_strategies = {name: float(share) for name, share in sorted_strategies[:3]}
        dominant = sorted_strategies[0][0]
        dominance = float(sorted_strategies[0][1])
    cost_history = [float(v) for v in history.get("system_cost", [])] if isinstance(history.get("system_cost", []), list) else []
    leakage_history = [float(v) for v in history.get("fraud_leakage", [])] if isinstance(history.get("fraud_leakage", []), list) else []
    mean_cost = float(np.mean(cost_history)) if cost_history else 0.0
    mean_leakage = float(np.mean(leakage_history)) if leakage_history else 0.0
    efficiency = float(1.0 / (1.0 + mean_cost + 0.1 * mean_leakage))

    # Variance-based classification using cost coefficient of variation (CV = std / mean).
    cost_arr = np.array(cost_history, dtype=float)
    cost_variance = float(np.var(cost_arr)) if len(cost_arr) >= 2 else 0.0
    cost_std = float(np.sqrt(cost_variance))
    cost_cv = float(cost_std / max(abs(mean_cost), 1.0))

    # Collect all individual strategy weights across history for variance.
    all_strategy_weights: List[float] = []
    if isinstance(strategy_history, list):
        for snapshot in strategy_history:
            if isinstance(snapshot, dict):
                all_strategy_weights.extend(float(v) for v in snapshot.values())
    strategy_variance = float(np.var(all_strategy_weights)) if all_strategy_weights else 0.0

    if len(cost_history) >= 4:
        cost_diff = np.diff(cost_arr)
        sign_changes = int(np.sum(np.sign(cost_diff[1:]) != np.sign(cost_diff[:-1])))
    else:
        sign_changes = 0
    divergence = bool(not latest_eq.get("equilibrium_reached", False) and len(cost_history) >= 2 and cost_history[-1] > cost_history[0])

    # Classify equilibrium type using CV thresholds and structural patterns.
    eq_reached = bool(latest_eq.get("equilibrium_reached", False))
    equilibrium_type: str
    if len(cost_history) < 2:
        equilibrium_type = "insufficient_data"
    elif eq_reached and cost_cv < EQUILIBRIUM_CV_STABLE_THRESHOLD:
        equilibrium_type = "stable_equilibrium"
    elif eq_reached:
        equilibrium_type = "oscillatory_equilibrium"
    elif sign_changes >= 2:
        equilibrium_type = "oscillatory"
    elif cost_cv > EQUILIBRIUM_CV_CHAOTIC_THRESHOLD:
        equilibrium_type = "chaotic_unstable"
    elif divergence:
        equilibrium_type = "divergent"
    else:
        equilibrium_type = "stable"

    # Keep behavior_pattern for backwards compatibility, mapped from equilibrium_type.
    _behavior_map = {
        "stable_equilibrium": "equilibrium_convergent",
        "oscillatory_equilibrium": "equilibrium_convergent",
    }
    behavior_pattern = _behavior_map.get(equilibrium_type, equilibrium_type)

    convergence_time = int(latest_eq.get("time_to_equilibrium", -1))
    equilibrium_state = {
        "reached": eq_reached,
        "time_to_equilibrium": convergence_time,
        "state": latest_eq.get("equilibrium_state", {}),
        "strategy_delta": float(latest_eq.get("strategy_delta", float("inf"))),
        "threshold_delta": float(latest_eq.get("threshold_delta", float("inf"))),
        "budget_delta": float(latest_eq.get("budget_delta", float("inf"))),
        "selection_delta": float(latest_eq.get("selection_delta", float("inf"))),
        "cost_delta_ratio": float(latest_eq.get("cost_delta_ratio", float("inf"))),
        "round": int(rounds[-1].iteration) if rounds else -1,
    }
    return {
        "equilibrium_reached": eq_reached,
        "equilibrium_type": equilibrium_type,
        "convergence_time": convergence_time,
        "convergence": eq_reached,
        "divergence": divergence,
        "behavior_pattern": behavior_pattern,
        "dominant_strategy": dominant,
        "dominance_share": dominance,
        "dominant_strategies": dominant_strategies,
        "equilibrium_state": equilibrium_state,
        "system_efficiency": efficiency,
        "fraud_prevented_pct": float(np.clip(100.0 - mean_leakage, 0.0, 100.0)),
        "mean_system_cost": mean_cost,
        "mean_fraud_leakage_pct": mean_leakage,
        "cost_variance": cost_variance,
        "strategy_variance": strategy_variance,
    }


def _scenario_for_iteration(iteration: int, scenario_schedule: List[str] | None = None) -> str:
    if scenario_schedule:
        return str(scenario_schedule[int(iteration) % len(scenario_schedule)])
    return ADVERSARIAL_SCENARIOS[int(iteration) % len(ADVERSARIAL_SCENARIOS)]


def _apply_scenario_pressure(params: SimulationParams, scenario_name: str) -> SimulationParams:
    out = dict(params)
    out_mix = dict(out.get("smart_strategy_mix", {}))
    config = SCENARIO_PRESSURE.get(scenario_name, {})
    for strategy_name in STRATEGY_NAMES:
        if strategy_name in config:
            out_mix[strategy_name] = float(out_mix.get(strategy_name, 0.0) + float(config[strategy_name]))
    if "noise_level" in config:
        out["noise_level"] = float(
            np.clip(float(out.get("noise_level", NOISE_BASE_LEVEL)) + float(config["noise_level"]), 0.01, 0.95)
        )
    if "delayed_observation_rate" in config:
        out["delayed_observation_rate"] = float(
            np.clip(
                float(out.get("delayed_observation_rate", DELAY_BASE_OBSERVATION_RATE))
                + float(config["delayed_observation_rate"]),
                0.01,
                0.60,
            )
        )
    if "temporal_drift_rate" in config:
        out["temporal_drift_rate"] = float(
            np.clip(
                float(out.get("temporal_drift_rate", BASE_TEMPORAL_DRIFT_RATE))
                + float(config["temporal_drift_rate"]),
                0.01,
                0.60,
            )
        )
    out["smart_strategy_mix"] = normalize_strategy_mix(out_mix)
    return cast(SimulationParams, out)


def _apply_environment_reactivity(
    params: SimulationParams,
    fraud_leakage_pct: float,
    system_cost: float,
    selected_ratio: float,
) -> SimulationParams:
    out = dict(params)
    leakage = float(np.clip(fraud_leakage_pct / 100.0, 0.0, 1.0))
    normalized_cost = float(np.clip(system_cost / 500.0, 0.0, 1.0))
    out["noise_level"] = float(np.clip(float(out.get("noise_level", NOISE_BASE_LEVEL)) + 0.10 * leakage, 0.01, 0.95))
    out["delayed_observation_rate"] = float(
        np.clip(
            float(out.get("delayed_observation_rate", DELAY_BASE_OBSERVATION_RATE))
            + 0.12 * leakage
            + 0.08 * normalized_cost,
            0.01,
            0.60,
        )
    )
    out["honest_anomaly_rate"] = float(
        np.clip(float(out.get("honest_anomaly_rate", NOISE_BASE_RATE)) + 0.10 * selected_ratio, 0.01, 0.60)
    )
    return cast(SimulationParams, out)


def _render_evolution_plots(history: Mapping[str, List[float] | List[Dict[str, float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_history = history.get("attacker_strategy_distribution", [])
    threshold_history = history.get("defender_threshold", [])
    cost_history = history.get("system_cost", [])
    leakage_history = history.get("fraud_leakage", [])
    equilibrium_detected_history = history.get("equilibrium_detected", [])
    equilibrium_strategy_delta = history.get("equilibrium_strategy_delta", [])
    equilibrium_cost_delta_ratio = history.get("equilibrium_cost_delta_ratio", [])
    if isinstance(strategy_history, list):
        fig, _ = plot_strategy_distribution_over_time(strategy_history)
        fig.savefig(output_dir / "strategy_distribution_over_time.png", dpi=220)
    if isinstance(threshold_history, list):
        fig, _ = plot_defender_threshold_over_time([float(v) for v in threshold_history])
        fig.savefig(output_dir / "defender_threshold_over_time.png", dpi=220)
    if isinstance(cost_history, list):
        fig, _ = plot_cost_vs_time([float(v) for v in cost_history])
        fig.savefig(output_dir / "cost_vs_time.png", dpi=220)
    if isinstance(leakage_history, list):
        fig, _ = plot_fraud_leakage_vs_time([float(v) for v in leakage_history])
        fig.savefig(output_dir / "fraud_leakage_vs_time.png", dpi=220)
    if (
        isinstance(equilibrium_detected_history, list)
        and isinstance(equilibrium_strategy_delta, list)
        and isinstance(equilibrium_cost_delta_ratio, list)
    ):
        fig, _ = plot_equilibrium_detection(
            equilibrium_detected_history=[float(v) for v in equilibrium_detected_history],
            strategy_delta_history=[float(v) for v in equilibrium_strategy_delta],
            cost_delta_ratio_history=[float(v) for v in equilibrium_cost_delta_ratio],
        )
        fig.savefig(output_dir / "equilibrium_detection_over_time.png", dpi=220)


class AdversarialFraudEnvironment:
    def __init__(
        self,
        iterations: int,
        seed: int,
        total_nodes: int,
        time_steps: int,
        attacker_memory_path: Path = ATTACKER_MEMORY_PATH,
        defender_memory_path: Path = DEFENDER_MEMORY_PATH,
        uncertainty_level: float = 0.08,
        inner_iterations: int = 3,
        inner_convergence_epsilon: float = INNER_CONVERGENCE_EPSILON,
        scenario_schedule: List[str] | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.total_nodes = int(total_nodes)
        self.time_steps = int(time_steps)
        self.rng = np.random.default_rng(seed)
        self.train_df = pd.DataFrame()
        self.final_model = None
        self.attacker_memory_path = Path(attacker_memory_path)
        self.defender_memory_path = Path(defender_memory_path)
        self.uncertainty_level = float(np.clip(uncertainty_level, 0.0, 0.5))
        self.inner_iterations = max(int(inner_iterations), 1)
        self.inner_convergence_epsilon = float(max(inner_convergence_epsilon, 0.0))
        self.scenario_schedule = list(scenario_schedule or [])
        # Tracks consecutive attacker strategy use for repetition penalty.
        self._last_chosen_strategy: str | None = None
        self._consecutive_strategy_count: int = 0
        self.attacker_memory = _load_json_memory(
            self.attacker_memory_path,
            {
                "strategy_success_rates": {},
                "strategy_outcomes": {},
                "failure_patterns": {},
                "run_count": 0,
            },
        )
        self.defender_memory = _load_json_memory(
            self.defender_memory_path,
            {
                "defender_effectiveness": {},
                "defender_responses": [],
                "run_count": 0,
            },
        )
        if not self.attacker_memory_path.exists() and not self.defender_memory_path.exists():
            legacy_memory = _load_json_memory(
                LEGACY_STRATEGIC_MEMORY_PATH,
                {
                    "strategy_success_rates": {},
                    "strategy_outcomes": {},
                    "failure_patterns": {},
                    "defender_effectiveness": {},
                    "defender_responses": [],
                    "run_count": 0,
                },
            )
            self.attacker_memory["strategy_success_rates"] = dict(legacy_memory.get("strategy_success_rates", {}))
            self.attacker_memory["strategy_outcomes"] = dict(legacy_memory.get("strategy_outcomes", {}))
            self.attacker_memory["failure_patterns"] = dict(legacy_memory.get("failure_patterns", {}))
            self.attacker_memory["run_count"] = int(legacy_memory.get("run_count", 0))
            self.defender_memory["defender_effectiveness"] = dict(legacy_memory.get("defender_effectiveness", {}))
            self.defender_memory["defender_responses"] = list(legacy_memory.get("defender_responses", []))
            self.defender_memory["run_count"] = int(legacy_memory.get("run_count", 0))
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
            "defender_budget_ratio": [],
            "defender_selection_strategy_id": [],
            "system_cost": [],
            "fraud_leakage": [],
            "equilibrium_detected": [],
            "equilibrium_strategy_delta": [],
            "equilibrium_cost_delta_ratio": [],
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
        scenario_name = _scenario_for_iteration(iteration, scenario_schedule=self.scenario_schedule)
        recent_leakage = float(self.history["fraud_leakage"][-1]) if self.history["fraud_leakage"] else 0.0
        recent_cost = float(self.history["system_cost"][-1]) if self.history["system_cost"] else 0.0
        defender_state = {
            "threshold": float(defender_agent.threshold),
            "budget_ratio": float(defender_agent.budget_ratio),
            "defender_intensity": float(np.clip(defender_agent.budget_ratio * DEFENDER_INTENSITY_SCALE, 0.0, 1.0)),
        }
        environment_state = {
            "fraud_leakage": float(np.clip(recent_leakage / 100.0, 0.0, 1.0)),
            "fraud_profit": float(
                np.clip(recent_leakage * FRAUD_PROFIT_FROM_LEAKAGE_SCALE, 0.0, FRAUD_PROFIT_CAP)
            ),
            "network_congestion": float(np.clip(self.sim_params["noise_level"], 0.0, 1.0)),
            "trust_shift": float(np.clip(1.0 - self.sim_params["honest_anomaly_rate"], -1.0, 1.0)),
            "defender_response_shift": float(np.clip(recent_cost / 1200.0, -0.2, 0.2)),
        }
        first_strategy, first_stage_simulations = attacker_agent.choose_strategic_strategy(
            defender_state=defender_state,
            environment_state=environment_state,
            rng=self.rng,
            foresight_steps=2,
            uncertainty_scale=0.08,
        )
        predicted_threshold = float(
            np.clip(first_stage_simulations[first_strategy].expected_defender_threshold, 0.2, 0.95)
        )
        defender_state["threshold"] = predicted_threshold
        chosen_strategy, strategic_simulations = attacker_agent.choose_strategic_strategy(
            defender_state=defender_state,
            environment_state=environment_state,
            rng=self.rng,
            foresight_steps=2,
            uncertainty_scale=0.10,
        )
        predicted_threshold = float(np.clip(strategic_simulations[chosen_strategy].expected_defender_threshold, 0.2, 0.95))
        acted_mix = dict(observation.strategy_mix)
        acted_mix[chosen_strategy] = float(acted_mix.get(chosen_strategy, 0.0) + SELECTED_STRATEGY_BOOST)
        acted_mix = normalize_strategy_mix(acted_mix)
        params = _normalize_sim_params(observation.simulation_params, strategy_mix=acted_mix)
        params = _apply_scenario_pressure(params, scenario_name=scenario_name)

        sim = build_network_simulation(
            total_nodes=self.total_nodes,
            time_steps=self.time_steps,
            seed=self.seed + iteration,
            difficulty_level=observation.difficulty_level,
            model_for_adaptation=self.final_model.model if self.final_model is not None else None,
            fraud_score_threshold=predicted_threshold,
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
            hard_cases = _hard_negative_mining(df, probs=pre_probs, threshold=predicted_threshold)
        else:
            hard_cases = df.iloc[0:0].copy()
        self.train_df = pd.concat([self.train_df, df, hard_cases], ignore_index=True)

        self.final_model = train_fraud_model(self.train_df, model_path="model.pkl", random_state=self.seed + iteration)
        probs = self.final_model.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        fraud_scores = {int(node_id): float(score) for node_id, score in zip(df["node_id"], probs)}
        label_lookup = {int(node_id): int(label) for node_id, label in zip(df["node_id"], df["label"])}
        uncertainty_values = 1.0 - np.abs((2.0 * probs) - 1.0)
        uncertainty_scores = {
            int(node_id): float(np.clip(u, 0.0, 1.0))
            for node_id, u in zip(df["node_id"], uncertainty_values)
        }
        observed_mix = {
            name: float(max(value + self.rng.normal(0.0, self.uncertainty_level), 0.0))
            for name, value in observation.strategy_mix.items()
        }
        noisy_attacker_estimate = normalize_strategy_mix(observed_mix) if observed_mix else dict(observation.strategy_mix)

        # True alternating best-response inner dynamics under imperfect information.
        best_response_mix = dict(noisy_attacker_estimate)
        best_response_strategy = chosen_strategy
        best_response_sims = strategic_simulations
        policy_plan: Dict[str, float | str] = {
            "threshold": float(defender_agent.threshold),
            "budget_ratio": float(defender_agent.budget_ratio),
            "selection_strategy": str(defender_agent.selection_strategy),
            "total_cost": float("inf"),
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "investigation_rate": 0.0,
        }
        defender_policy_guess = {
            "threshold": float(defender_agent.threshold),
            "budget_ratio": float(defender_agent.budget_ratio),
            "selection_strategy": str(defender_agent.selection_strategy),
        }
        # Iterative alternating best-response with epsilon-convergence stopping.
        prev_inner_threshold = defender_policy_guess["threshold"]
        prev_inner_budget = defender_policy_guess["budget_ratio"]
        for _ in range(self.inner_iterations):
            # Attacker: best-responds to noisy estimate of defender policy.
            estimated_threshold = float(
                np.clip(
                    defender_policy_guess["threshold"] + self.rng.normal(0.0, self.uncertainty_level),
                    0.2,
                    0.95,
                )
            )
            estimated_budget_ratio = float(
                np.clip(
                    defender_policy_guess["budget_ratio"] + self.rng.normal(0.0, self.uncertainty_level * 0.6),
                    0.0,
                    0.6,
                )
            )
            attacker_defender_state = {
                "threshold": estimated_threshold,
                "budget_ratio": estimated_budget_ratio,
                "defender_intensity": float(np.clip(estimated_budget_ratio * DEFENDER_INTENSITY_SCALE, 0.0, 1.0)),
            }
            best_response_strategy, best_response_sims = attacker_agent.choose_strategic_strategy(
                defender_state=attacker_defender_state,
                environment_state=environment_state,
                rng=self.rng,
                foresight_steps=2,
                uncertainty_scale=self.uncertainty_level,
            )
            best_response_mix = dict(attacker_agent.strategy_mix)
            best_response_mix[best_response_strategy] = float(best_response_mix.get(best_response_strategy, 0.0) + SELECTED_STRATEGY_BOOST)
            best_response_mix = normalize_strategy_mix(best_response_mix)
            # Defender: observes noisy mix and samples belief via Dirichlet (Thompson sampling).
            defender_observed_mix = {
                name: float(max(value + self.rng.normal(0.0, self.uncertainty_level * 0.5), 0.0))
                for name, value in best_response_mix.items()
            }
            canonical_mix = normalize_strategy_mix(defender_observed_mix)
            alpha = np.array(
                [float(canonical_mix[name]) * DIRICHLET_CONCENTRATION + 0.1 for name in sorted(canonical_mix)],
                dtype=float,
            )
            sampled_probs = self.rng.dirichlet(alpha)
            sampled_attacker_estimate = dict(zip(sorted(canonical_mix.keys()), sampled_probs.tolist()))
            policy_plan = defender_agent.optimize_policy(
                fraud_scores=fraud_scores,
                labels=label_lookup,
                uncertainty_scores=uncertainty_scores,
                attacker_strategy_estimate=sampled_attacker_estimate,
                grid_size=17,
            )
            defender_policy_guess = {
                "threshold": float(policy_plan["threshold"]),
                "budget_ratio": float(policy_plan["budget_ratio"]),
                "selection_strategy": str(policy_plan["selection_strategy"]),
            }
            # Convergence check: stop inner loop when policy change is negligible.
            threshold_delta_inner = abs(float(policy_plan["threshold"]) - prev_inner_threshold)
            budget_delta_inner = abs(float(policy_plan["budget_ratio"]) - prev_inner_budget)
            if max(threshold_delta_inner, budget_delta_inner) < self.inner_convergence_epsilon:
                break
            prev_inner_threshold = float(policy_plan["threshold"])
            prev_inner_budget = float(policy_plan["budget_ratio"])
        chosen_strategy = best_response_strategy
        strategic_simulations = best_response_sims
        defender_agent.threshold = float(policy_plan["threshold"])
        defender_agent.budget_ratio = float(policy_plan["budget_ratio"])
        defender_agent.selection_strategy = str(policy_plan["selection_strategy"])
        selected_nodes = defender_agent.select_nodes(fraud_scores=fraud_scores, uncertainty_scores=uncertainty_scores)

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
        if "region" in df.columns:
            node_regions = {int(node_id): str(region) for node_id, region in zip(df["node_id"], df["region"])}
        else:
            node_regions = {int(node_id): "unknown" for node_id in df["node_id"].tolist()}
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
        strategic_outcome = strategic_simulations.get(chosen_strategy)
        if strategic_outcome is not None and chosen_strategy in strategy_feedback:
            strategy_feedback[chosen_strategy]["economic_gain"] += float(max(strategic_outcome.expected_gain, 0.0))
            strategy_feedback[chosen_strategy]["detection_penalty"] += float(max(strategic_outcome.expected_penalty, 0.0))

        # Environment pressure: penalize attacker for repeating the same strategy.
        if chosen_strategy == self._last_chosen_strategy:
            self._consecutive_strategy_count += 1
        else:
            self._consecutive_strategy_count = 1
        self._last_chosen_strategy = chosen_strategy
        if self._consecutive_strategy_count >= 2 and chosen_strategy in strategy_feedback:
            repeat_penalty = float(min(STRATEGY_REPEAT_PENALTY_SCALE * self._consecutive_strategy_count, STRATEGY_REPEAT_PENALTY_CAP))
            strategy_feedback[chosen_strategy]["detection_penalty"] = (
                float(strategy_feedback[chosen_strategy].get("detection_penalty", 0.0)) + repeat_penalty
            )

        # Environment pressure: harden attacker rewards when leakage is high (system adapts).
        recent_leakage_frac = float(np.clip(economic_summary.get("pct_reward_lost_to_fraud", 0.0) / 100.0, 0.0, 1.0))
        if recent_leakage_frac > LEAKAGE_HARDENING_THRESHOLD:
            hardening_factor = float(np.clip(1.0 - LEAKAGE_HARDENING_SCALE * (recent_leakage_frac - LEAKAGE_HARDENING_THRESHOLD), 0.4, 1.0))
            for s in strategy_feedback:
                strategy_feedback[s]["economic_gain"] = float(strategy_feedback[s].get("economic_gain", 0.0)) * hardening_factor

        strategy_mix = attacker_agent.update(strategy_feedback=strategy_feedback)
        defender_agent.update_policy(
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            delayed_detection_rate=float(detection_metrics.summary.get("avg_detection_delay", 0.0)),
            system_cost=float(system_costs.get("total_system_cost", 0.0)),
        )

        failure_driven = _compute_augmentation_params(failures)
        self.sim_params = _normalize_sim_params(failure_driven, strategy_mix=strategy_mix)
        selected_ratio = float(len(selected_nodes) / max(len(df), 1))
        self.sim_params = _apply_environment_reactivity(
            self.sim_params,
            fraud_leakage_pct=float(economic_summary.get("pct_reward_lost_to_fraud", 0.0)),
            system_cost=float(system_costs.get("total_system_cost", 0.0)),
            selected_ratio=selected_ratio,
        )

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
                "optimized_threshold_cost": float(policy_plan.get("total_cost", 0.0)),
                "optimized_threshold_fp_rate": float(policy_plan.get("false_positive_rate", 0.0)),
                "optimized_threshold_fn_rate": float(policy_plan.get("false_negative_rate", 0.0)),
                "optimized_investigation_rate": float(policy_plan.get("investigation_rate", 0.0)),
                "strategic_expected_gain": float(strategic_simulations[chosen_strategy].expected_gain),
                "strategic_expected_penalty": float(strategic_simulations[chosen_strategy].expected_penalty),
                "strategic_expected_utility": float(strategic_simulations[chosen_strategy].expected_utility),
            }
        )

        self.history["attacker_strategy_distribution"].append(dict(strategy_mix))
        self.history["defender_threshold"].append(float(defender_agent.threshold))
        self.history["defender_budget_ratio"].append(float(defender_agent.budget_ratio))
        selection_to_id = {"risk_only": 0.0, "risk_adjusted": 1.0, "uncertainty_first": 2.0}
        self.history["defender_selection_strategy_id"].append(
            float(selection_to_id.get(str(defender_agent.selection_strategy), 1.0))
        )
        self.history["system_cost"].append(float(system_costs.get("total_system_cost", 0.0)))
        self.history["fraud_leakage"].append(float(economic_summary.get("pct_reward_lost_to_fraud", 0.0)))
        equilibrium = detect_equilibrium(self.history)
        self.history["equilibrium_detected"].append(1.0 if bool(equilibrium["is_equilibrium"]) else 0.0)
        self.history["equilibrium_strategy_delta"].append(float(equilibrium["strategy_delta"]))
        self.history["equilibrium_cost_delta_ratio"].append(float(equilibrium["cost_delta_ratio"]))

        strategy_memory = self.attacker_memory.setdefault("strategy_success_rates", {})
        if isinstance(strategy_memory, dict):
            stats = strategy_memory.setdefault(chosen_strategy, {"count": 0, "avg_utility": 0.0})
            if isinstance(stats, dict):
                prev_count = int(stats.get("count", 0))
                prev_avg = float(stats.get("avg_utility", 0.0))
                next_count = prev_count + 1
                utility = float(strategic_simulations[chosen_strategy].expected_utility)
                next_avg = prev_avg + (utility - prev_avg) / max(next_count, 1)
                stats["count"] = int(next_count)
                stats["avg_utility"] = float(next_avg)
        outcome_memory = self.attacker_memory.setdefault("strategy_outcomes", {})
        if isinstance(outcome_memory, dict):
            utility = float(strategic_simulations[chosen_strategy].expected_utility)
            stats = outcome_memory.setdefault(chosen_strategy, {"success_count": 0, "failure_count": 0})
            if isinstance(stats, dict):
                if utility >= 0.0:
                    stats["success_count"] = int(stats.get("success_count", 0)) + 1
                else:
                    stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
        failure_memory = self.attacker_memory.setdefault("failure_patterns", {})
        if isinstance(failure_memory, dict):
            for key, value in failures.items():
                failure_memory[key] = int(failure_memory.get(key, 0)) + int(value)
        defender_memory = self.defender_memory.setdefault("defender_effectiveness", {})
        if isinstance(defender_memory, dict):
            prev_runs = int(defender_memory.get("count", 0))
            prev_cost = float(defender_memory.get("avg_cost", 0.0))
            prev_threshold = float(defender_memory.get("avg_threshold", 0.5))
            next_runs = prev_runs + 1
            next_cost = prev_cost + (float(system_costs.get("total_system_cost", 0.0)) - prev_cost) / max(next_runs, 1)
            next_threshold = prev_threshold + (float(defender_agent.threshold) - prev_threshold) / max(next_runs, 1)
            defender_memory["count"] = int(next_runs)
            defender_memory["avg_cost"] = float(next_cost)
            defender_memory["avg_threshold"] = float(next_threshold)
            defender_memory["avg_budget_ratio"] = float(
                defender_memory.get("avg_budget_ratio", defender_agent.budget_ratio)
                + (float(defender_agent.budget_ratio) - float(defender_memory.get("avg_budget_ratio", defender_agent.budget_ratio)))
                / max(next_runs, 1)
            )
        defender_responses = self.defender_memory.setdefault("defender_responses", [])
        if isinstance(defender_responses, list):
            defender_responses.append(
                {
                    "threshold": float(defender_agent.threshold),
                    "budget_ratio": float(defender_agent.budget_ratio),
                    "selection_strategy": str(defender_agent.selection_strategy),
                    "system_cost": float(system_costs.get("total_system_cost", 0.0)),
                }
            )
            if len(defender_responses) > 250:
                del defender_responses[:-250]

        iteration_result = ClosedLoopIteration(
            iteration=iteration,
            difficulty_level=observation.difficulty_level,
            hard_negative_count=int(len(hard_cases)),
            metrics=round_metrics,
            selected_nodes=selected_nodes,
            failure_breakdown=failures,
            attacker_strategy_mix=dict(strategy_mix),
            defender_threshold=float(defender_agent.threshold),
            defender_budget_ratio=float(defender_agent.budget_ratio),
            defender_selection_strategy=str(defender_agent.selection_strategy),
            selected_attacker_strategy=chosen_strategy,
            selected_adversarial_scenario=scenario_name,
            equilibrium_detected=bool(equilibrium["is_equilibrium"]),
        )
        next_observation = (
            self._observation(iteration=iteration + 1, attacker=attacker_agent, defender=defender_agent)
            if (iteration + 1) < self.iterations
            else None
        )
        _save_json_memory(self.attacker_memory_path, self.attacker_memory)
        _save_json_memory(self.defender_memory_path, self.defender_memory)
        return StepOutcome(result=iteration_result, next_observation=next_observation)


def run_closed_loop(
    iterations: int = 3,
    seed: int = 42,
    total_nodes: int = 220,
    time_steps: int = 18,
    output_dir: str | None = None,
    memory_path: str | None = None,
    attacker_memory_path: str | None = None,
    defender_memory_path: str | None = None,
    uncertainty_level: float = 0.08,
    inner_iterations: int = 3,
    inner_convergence_epsilon: float = INNER_CONVERGENCE_EPSILON,
    memory_decay_rate: float = MEMORY_DECAY_RATE,
    scenario_schedule: List[str] | None = None,
) -> ClosedLoopArtifacts:
    resolved_attacker_memory_path = Path(attacker_memory_path) if attacker_memory_path else ATTACKER_MEMORY_PATH
    resolved_defender_memory_path = Path(defender_memory_path) if defender_memory_path else DEFENDER_MEMORY_PATH
    if memory_path:
        base = Path(memory_path)
        if base.suffix == ".json":
            parent = base.parent if base.parent != Path(".") else MEMORY_DIR
            resolved_attacker_memory_path = parent / "attacker_memory.json"
            resolved_defender_memory_path = parent / "defender_memory.json"
        else:
            resolved_attacker_memory_path = base / "attacker_memory.json"
            resolved_defender_memory_path = base / "defender_memory.json"
    environment = AdversarialFraudEnvironment(
        iterations=iterations,
        seed=seed,
        total_nodes=total_nodes,
        time_steps=time_steps,
        attacker_memory_path=resolved_attacker_memory_path,
        defender_memory_path=resolved_defender_memory_path,
        uncertainty_level=uncertainty_level,
        inner_iterations=inner_iterations,
        inner_convergence_epsilon=inner_convergence_epsilon,
        scenario_schedule=scenario_schedule,
    )
    remembered_strategy = environment.attacker_memory.get("strategy_success_rates", {})
    remembered_outcomes = environment.attacker_memory.get("strategy_outcomes", {})
    remembered_defense = environment.defender_memory.get("defender_effectiveness", {})
    remembered_responses = environment.defender_memory.get("defender_responses", [])
    attacker = AttackerAgent(temperature=0.25, epsilon=0.08, use_ucb_selection=True)
    if isinstance(remembered_strategy, dict):
        # Apply exponential decay so older memory has less influence on the warm-start.
        run_count = int(environment.attacker_memory.get("run_count", 0))
        decay_multiplier = float(float(memory_decay_rate) ** min(run_count, 20))
        warm_rewards: Dict[str, float] = {}
        for strategy_name, stats in remembered_strategy.items():
            if isinstance(stats, dict):
                avg_utility = float(stats.get("avg_utility", 0.0)) * decay_multiplier
                warm_rewards[str(strategy_name)] = avg_utility
        if isinstance(remembered_outcomes, dict):
            for strategy_name, stats in remembered_outcomes.items():
                if isinstance(stats, dict):
                    success = float(stats.get("success_count", 0))
                    failure = float(stats.get("failure_count", 0))
                    total_outcomes = success + failure
                    if total_outcomes > 0:
                        # Weight by success rate centred at 0.5, scaled and decayed.
                        success_rate_bias = MEMORY_SUCCESS_RATE_BIAS_SCALE * (success / total_outcomes - 0.5) * decay_multiplier
                    else:
                        success_rate_bias = 0.0
                    warm_rewards[str(strategy_name)] = float(warm_rewards.get(str(strategy_name), 0.0) + success_rate_bias)
        if warm_rewards:
            attacker.update(rewards=warm_rewards)
    defender_threshold = 0.5
    defender_budget_ratio = 0.07
    defender_selection_strategy = "risk_adjusted"
    if isinstance(remembered_defense, dict):
        defender_threshold = float(np.clip(float(remembered_defense.get("avg_threshold", 0.5)), 0.2, 0.95))
        defender_budget_ratio = float(np.clip(float(remembered_defense.get("avg_budget_ratio", 0.07)), 0.01, 0.40))
    if isinstance(remembered_responses, list) and remembered_responses:
        last = remembered_responses[-1]
        if isinstance(last, dict):
            defender_selection_strategy = str(last.get("selection_strategy", defender_selection_strategy))
            defender_budget_ratio = float(np.clip(float(last.get("budget_ratio", defender_budget_ratio)), 0.01, 0.40))
    defender = DefenderAgent(
        threshold=defender_threshold,
        budget_ratio=defender_budget_ratio,
        selection_strategy=defender_selection_strategy,
        min_budget=3,
    )

    rounds: List[ClosedLoopIteration] = []
    for i in range(iterations):
        outcome = environment.step(iteration=i, attacker_agent=attacker, defender_agent=defender)
        rounds.append(outcome.result)

    environment.attacker_memory["run_count"] = int(environment.attacker_memory.get("run_count", 0)) + 1
    environment.defender_memory["run_count"] = int(environment.defender_memory.get("run_count", 0)) + 1
    _save_json_memory(resolved_attacker_memory_path, environment.attacker_memory)
    _save_json_memory(resolved_defender_memory_path, environment.defender_memory)
    if output_dir:
        _render_evolution_plots(environment.history, output_dir=Path(output_dir))
    benchmark = benchmark_fraud_models(train_dataset=environment.train_df, test_dataset=environment.train_df, random_state=seed)
    system_analysis = analyze_system_dynamics(environment.history, rounds=rounds)
    return ClosedLoopArtifacts(
        iterations=rounds,
        final_model=environment.final_model,
        benchmark=benchmark,
        last_features=environment.train_df,
        simulation_params=dict(environment.sim_params),
        evolution_history=dict(environment.history),
        system_analysis=system_analysis,
    )
