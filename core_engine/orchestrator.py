from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping
import pandas as pd

from core_engine.loop import ClosedLoopArtifacts, ClosedLoopIteration, run_closed_loop
from features import extract_node_features
from simulation import build_network_simulation
from utils.constants import FEATURE_COLUMNS


@dataclass(frozen=True)
class PipelineArtifacts:
    model_artifacts: object
    benchmark: pd.DataFrame
    features: pd.DataFrame
    failure_analysis: pd.DataFrame
    advanced_metrics: dict


@dataclass(frozen=True)
class ExperimentResult:
    runs: List[ClosedLoopArtifacts]
    run_summaries: pd.DataFrame
    comparison: Dict[str, float]


def run_adversarial_stress_tests(model_artifacts, seed: int = 142) -> pd.DataFrame:
    rows = []
    for idx, noise_level in enumerate([0.03, 0.08, 0.14, 0.20]):
        sim = build_network_simulation(seed=seed + idx, noise_level=noise_level)
        df = extract_node_features(sim)
        probs = model_artifacts.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        pred = (probs >= model_artifacts.threshold).astype(int)
        y = df["label"].astype(int)
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        rows.append({"noise_level": float(noise_level), "false_positives": fp, "false_negatives": fn})
    return pd.DataFrame(rows)


def run_pipeline(seed: int = 42, iterations: int = 3) -> PipelineArtifacts:
    result = run_closed_loop(iterations=iterations, seed=seed)
    final_iteration = result.iterations[-1]
    failure_analysis = pd.DataFrame([final_iteration.failure_breakdown])
    advanced_metrics = {
        "defender_threshold": float(final_iteration.defender_threshold),
        "attacker_strategy_dominance": float(max(final_iteration.attacker_strategy_mix.values())),
        "system_cost_trend_final": float(final_iteration.metrics.get("total_system_cost", 0.0)),
        "fraud_leakage_final": float(final_iteration.metrics.get("fraud_leakage_pct", 0.0)),
    }
    return PipelineArtifacts(
        model_artifacts=result.final_model,
        benchmark=result.benchmark,
        features=result.last_features,
        failure_analysis=failure_analysis,
        advanced_metrics=advanced_metrics,
    )


def run_experiment(config: Mapping[str, object]) -> ExperimentResult:
    runs = max(int(config.get("runs", 1)), 1)
    iterations = max(int(config.get("iterations", 4)), 1)
    seed = int(config.get("seed", 42))
    total_nodes = max(int(config.get("total_nodes", 220)), 20)
    time_steps = max(int(config.get("time_steps", 18)), 2)

    artifacts: List[ClosedLoopArtifacts] = []
    summary_rows: List[Dict[str, object]] = []
    for run_idx in range(runs):
        result = run_closed_loop(
            iterations=iterations,
            seed=seed + run_idx,
            total_nodes=total_nodes,
            time_steps=time_steps,
        )
        artifacts.append(result)
        final_iteration = result.iterations[-1]
        summary_rows.append(
            {
                "run_id": int(run_idx),
                "seed": int(seed + run_idx),
                "difficulty_final": str(final_iteration.difficulty_level),
                "system_cost_final": float(final_iteration.metrics.get("total_system_cost", 0.0)),
                "fraud_leakage_final": float(final_iteration.metrics.get("fraud_leakage_pct", 0.0)),
                "defender_threshold_final": float(final_iteration.defender_threshold),
                "attacker_strategy_dominance": float(max(final_iteration.attacker_strategy_mix.values())),
                "avg_detection_delay_final": float(final_iteration.metrics.get("avg_detection_delay", 0.0)),
            }
        )

    run_summaries = pd.DataFrame(summary_rows)
    comparison = {
        "mean_system_cost": float(run_summaries["system_cost_final"].mean()) if not run_summaries.empty else 0.0,
        "std_system_cost": float(run_summaries["system_cost_final"].std(ddof=0)) if not run_summaries.empty else 0.0,
        "mean_fraud_leakage": float(run_summaries["fraud_leakage_final"].mean()) if not run_summaries.empty else 0.0,
        "mean_defender_threshold": float(run_summaries["defender_threshold_final"].mean()) if not run_summaries.empty else 0.0,
        "best_run_by_cost": float(run_summaries["system_cost_final"].idxmin()) if not run_summaries.empty else -1.0,
    }
    return ExperimentResult(runs=artifacts, run_summaries=run_summaries, comparison=comparison)


__all__ = [
    "ClosedLoopArtifacts",
    "ClosedLoopIteration",
    "PipelineArtifacts",
    "ExperimentResult",
    "run_adversarial_stress_tests",
    "run_closed_loop",
    "run_pipeline",
    "run_experiment",
]


if __name__ == "__main__":
    artifacts = run_pipeline(seed=42)
    print("selected model:", artifacts.model_artifacts.selected_model_name)
    print("metrics:", artifacts.model_artifacts.metrics)
    print("advanced_metrics:", artifacts.advanced_metrics)
    experiment = run_experiment({"runs": 2, "iterations": 3, "seed": 42})
    print("experiment comparison:", experiment.comparison)
