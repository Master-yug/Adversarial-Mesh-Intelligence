from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from constants import FEATURE_COLUMNS
from detection_metrics import compute_detection_metrics
from economic import simulate_economic_rewards
from features import extract_node_features, extract_temporal_node_features, feature_meanings_dataframe
from modeling import train_fraud_model
from simulation import build_network_simulation
from visualization import (
    plot_detection_over_time,
    plot_fraud_score_distribution,
    plot_network,
    plot_reward_distribution,
    plot_roc_curve,
)

HONEST_BINARY_LABEL = 0
ATTACKER_BINARY_LABEL = 1


@dataclass(frozen=True)
class PipelineArtifacts:
    base_simulation: object
    adaptive_simulation: object
    temporal_features: pd.DataFrame
    features: pd.DataFrame
    model_artifacts: object
    feature_meanings: pd.DataFrame
    detection_metrics: object
    economic_metrics: object


def run_adversarial_stress_tests(model_artifacts, seed: int = 142) -> pd.DataFrame:
    stress_scenarios = [
        {
            "name": "smart_attackers_mimic_honest",
            "kwargs": {
                "seed": seed,
                "honest_anomaly_rate": 0.10,
                "measurement_error_std": 0.07,
                "packet_loss_rate": 0.05,
                "missing_latency_rate": 0.05,
            },
        },
        {
            "name": "high_latency_honest_regions",
            "kwargs": {
                "seed": seed + 1,
                "honest_anomaly_rate": 0.18,
                "honest_unstable_rate": 0.15,
                "measurement_error_std": 0.08,
            },
        },
        {
            "name": "sparse_connectivity_partial_visibility",
            "kwargs": {
                "seed": seed + 2,
                "partial_visibility_rate": 0.22,
                "packet_loss_rate": 0.10,
                "missing_latency_rate": 0.08,
            },
        },
        {
            "name": "low_and_slow_strategy_dominant",
            "kwargs": {
                "seed": seed + 3,
                "smart_strategy_mix": {"low_and_slow": 0.85, "burst_attack": 0.1, "camouflage": 0.05},
            },
        },
        {
            "name": "burst_attack_strategy_dominant",
            "kwargs": {
                "seed": seed + 4,
                "smart_strategy_mix": {"low_and_slow": 0.05, "burst_attack": 0.85, "camouflage": 0.10},
            },
        },
        {
            "name": "camouflage_strategy_dominant",
            "kwargs": {
                "seed": seed + 5,
                "smart_strategy_mix": {"low_and_slow": 0.05, "burst_attack": 0.1, "camouflage": 0.85},
            },
        },
    ]

    rows = []
    for scenario in stress_scenarios:
        sim = build_network_simulation(**scenario["kwargs"])
        df = extract_node_features(sim)
        X = df[FEATURE_COLUMNS]
        y = df["label"].astype(int)
        y_prob = model_artifacts.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= model_artifacts.threshold).astype(int)

        false_positives = int(((y == HONEST_BINARY_LABEL) & (y_pred == ATTACKER_BINARY_LABEL)).sum())
        false_negatives = int(((y == ATTACKER_BINARY_LABEL) & (y_pred == HONEST_BINARY_LABEL)).sum())
        rows.append(
            {
                "scenario": scenario["name"],
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "fp_rate": float(false_positives / max(int((y == HONEST_BINARY_LABEL).sum()), 1)),
                "fn_rate": float(false_negatives / max(int((y == ATTACKER_BINARY_LABEL).sum()), 1)),
            }
        )

    return pd.DataFrame(rows)


def run_pipeline(seed: int = 42) -> PipelineArtifacts:
    base_simulation = build_network_simulation(seed=seed, time_steps=20)
    temporal_features = extract_temporal_node_features(base_simulation)
    features_df = extract_node_features(base_simulation)
    model_artifacts = train_fraud_model(features_df, model_path="model.pkl", random_state=seed)
    meanings = feature_meanings_dataframe()

    adaptive_simulation = build_network_simulation(
        seed=seed + 7,
        time_steps=20,
        model_for_adaptation=model_artifacts.model,
        fraud_score_threshold=model_artifacts.threshold,
        feature_columns=FEATURE_COLUMNS,
    )
    adaptive_features_df = extract_node_features(adaptive_simulation)
    adaptive_probs = model_artifacts.model.predict_proba(adaptive_features_df[FEATURE_COLUMNS])[:, 1]
    adaptive_pred = (adaptive_probs >= model_artifacts.threshold).astype(int)
    predicted_attacker_ids = adaptive_features_df.loc[
        adaptive_pred == 1,
        "node_id",
    ].tolist()

    node_labels = {node.node_id: node.label for node in adaptive_simulation.nodes}
    node_regions = {node.node_id: node.region for node in adaptive_simulation.nodes}
    high_value_region_map = {"new_york": 1.4, "tokyo": 1.5, "london": 1.35, "singapore": 1.45}
    connectivity_over_time = []
    for snapshot in adaptive_simulation.time_steps:
        graph = snapshot.peer_graph
        max_degree = max((graph.degree(node_id) for node_id in graph.nodes), default=1)
        connectivity_over_time.append(
            {
                int(node_id): float(graph.degree(node_id) / max(max_degree, 1))
                for node_id in graph.nodes
            }
        )
    detection_result = compute_detection_metrics(
        fraud_scores_over_time=adaptive_simulation.fraud_scores_over_time,
        node_labels=node_labels,
        threshold=model_artifacts.threshold,
        within_steps=5,
        node_regions=node_regions,
        high_value_regions=set(high_value_region_map),
    )
    economic_result = simulate_economic_rewards(
        fraud_scores_over_time=adaptive_simulation.fraud_scores_over_time,
        node_labels=node_labels,
        threshold=model_artifacts.threshold,
        node_regions=node_regions,
        connectivity_over_time=connectivity_over_time,
        high_value_regions=high_value_region_map,
    )

    plot_network(adaptive_simulation, detected_fraud_nodes=predicted_attacker_ids, output_path="network.png")
    plot_fraud_score_distribution(
        fraud_scores=adaptive_probs,
        labels=adaptive_features_df["label"].to_numpy(dtype=int),
        output_path="fraud_score_distribution.png",
    )
    plot_roc_curve(
        y_true=adaptive_features_df["label"].to_numpy(dtype=int),
        y_scores=adaptive_probs,
        output_path="roc_curve.png",
    )
    plot_detection_over_time(
        false_positives_over_time=detection_result.false_positives_over_time,
        output_path="detection_over_time.png",
    )
    plot_reward_distribution(
        per_node_rewards=economic_result.per_node_rewards,
        output_path="reward_distribution.png",
    )

    return PipelineArtifacts(
        base_simulation=base_simulation,
        adaptive_simulation=adaptive_simulation,
        temporal_features=temporal_features,
        features=features_df,
        model_artifacts=model_artifacts,
        feature_meanings=meanings,
        detection_metrics=detection_result,
        economic_metrics=economic_result,
    )


if __name__ == "__main__":
    artifacts = run_pipeline(seed=42)
    stress_df = run_adversarial_stress_tests(artifacts.model_artifacts, seed=99)
    print(artifacts.features.head())
    print("selected model:", artifacts.model_artifacts.selected_model_name)
    print("metrics:", artifacts.model_artifacts.metrics)
    print("confusion matrix:\n", artifacts.model_artifacts.confusion_matrix)
    print("model comparison:\n", artifacts.model_artifacts.model_comparison)
    print("feature importance:\n", artifacts.model_artifacts.feature_importance.head(10))
    print("feature meanings:\n", artifacts.feature_meanings.head(10))
    print("detection summary:\n", artifacts.detection_metrics.summary)
    print("economic summary:\n", artifacts.economic_metrics.summary)
    print("scenario summary:\n", artifacts.adaptive_simulation.scenario_metrics)
    print("stress-test summary:\n", stress_df)
