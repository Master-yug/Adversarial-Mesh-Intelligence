from __future__ import annotations

from dataclasses import dataclass

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
    }
    return PipelineArtifacts(
        model_artifacts=result.final_model,
        benchmark=result.benchmark,
        features=result.last_features,
        failure_analysis=failure_analysis,
        advanced_metrics=advanced_metrics,
    )


__all__ = [
    "ClosedLoopArtifacts",
    "ClosedLoopIteration",
    "PipelineArtifacts",
    "run_adversarial_stress_tests",
    "run_closed_loop",
    "run_pipeline",
]


if __name__ == "__main__":
    artifacts = run_pipeline(seed=42)
    print("selected model:", artifacts.model_artifacts.selected_model_name)
    print("metrics:", artifacts.model_artifacts.metrics)
    print("advanced_metrics:", artifacts.advanced_metrics)
