from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np
import pandas as pd

from utils.constants import FEATURE_COLUMNS
from evaluation.metrics import compute_detection_metrics, compute_system_cost_metrics
from evaluation.economic import simulate_economic_rewards
from features import extract_node_features
from simulation import build_network_simulation


class FraudModelPlugin(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class BenchmarkResult:
    system_cost: Dict[str, float]
    robustness_score: float
    failure_breakdown: Dict[str, int]
    attacker_roi: float
    defender_roi: float


def run_benchmark_service(
    plugin_model: FraudModelPlugin,
    threshold: float = 0.5,
    seed: int = 123,
) -> BenchmarkResult:
    sim = build_network_simulation(
        seed=seed,
        difficulty_level="hard",
        model_for_adaptation=plugin_model,
        feature_columns=FEATURE_COLUMNS,
        fraud_score_threshold=threshold,
    )
    df = extract_node_features(sim)
    probs = plugin_model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    preds = (probs >= threshold).astype(int)
    y_true = df["label"].astype(int).to_numpy()
    failure_breakdown = {
        "false_positive": int(np.sum((y_true == 0) & (preds == 1))),
        "false_negative": int(np.sum((y_true == 1) & (preds == 0))),
        "true_positive": int(np.sum((y_true == 1) & (preds == 1))),
        "true_negative": int(np.sum((y_true == 0) & (preds == 0))),
    }
    node_labels = {n.node_id: n.label for n in sim.nodes}
    node_regions = {n.node_id: n.region for n in sim.nodes}
    detection = compute_detection_metrics(
        fraud_scores_over_time=sim.fraud_scores_over_time,
        node_labels=node_labels,
        threshold=threshold,
        node_regions=node_regions,
        high_value_regions={"new_york", "tokyo", "london", "singapore"},
    )
    system_cost = compute_system_cost_metrics(
        per_node=detection.per_node,
        false_positives_over_time=detection.false_positives_over_time,
    )
    economic = simulate_economic_rewards(
        fraud_scores_over_time=sim.fraud_scores_over_time,
        node_labels=node_labels,
        threshold=threshold,
        node_regions=node_regions,
    )
    attacker_profit = float(economic.summary.get("total_fraud_profit", 0.0))
    defender_cost = float(system_cost.get("total_system_cost", 0.0))
    defender_gain = float(economic.summary.get("fraud_reduction_after_detection", 0.0))
    robustness_score = float(
        max(
            0.0,
            1.0 - (failure_breakdown["false_positive"] + failure_breakdown["false_negative"]) / max(len(df), 1),
        )
    )
    return BenchmarkResult(
        system_cost=system_cost,
        robustness_score=robustness_score,
        failure_breakdown=failure_breakdown,
        attacker_roi=float(attacker_profit / max(defender_cost, 1e-9)),
        defender_roi=float(defender_gain / max(defender_cost, 1e-9)),
    )
