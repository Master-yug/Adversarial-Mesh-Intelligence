from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import pandas as pd

from constants import FEATURE_COLUMNS
from detection_metrics import compute_detection_metrics
from features import extract_node_features
from simulation import build_network_simulation
from visualization import (
    plot_detection_over_time,
    plot_fraud_score_distribution,
    plot_network,
    plot_reward_distribution,
    plot_roc_curve,
)

try:
    from simulation import run_adaptive_simulation  # type: ignore[attr-defined]
except ImportError:
    run_adaptive_simulation = None  # type: ignore[assignment]

try:
    from modeling import train_model  # type: ignore[attr-defined]
except ImportError:
    from modeling import train_fraud_model as train_model  # type: ignore[assignment]

try:
    from economic import simulate_economic_rewards
except ImportError:
    simulate_economic_rewards = None  # type: ignore[assignment]


def _call_with_supported_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    signature = inspect.signature(func)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(**supported)


def parse_args() -> argparse.Namespace:
    """Parse demo CLI arguments."""
    parser = argparse.ArgumentParser(description="Decentralized anti-spoofing demo run")
    parser.add_argument("--nodes", type=int, default=200, help="Total number of simulated nodes")
    parser.add_argument("--timesteps", type=int, default=20, help="Number of adaptive simulation timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="Directory for output figures and model artifact",
    )
    return parser.parse_args()


def run_simulation(nodes: int, timesteps: int, seed: int):
    """Run baseline simulation and print core population counts."""
    logging.info("Step 1/5: Running baseline simulation...")
    simulation = build_network_simulation(total_nodes=nodes, time_steps=timesteps, seed=seed)
    attacker_count = sum(1 for node in simulation.nodes if node.label != "honest")
    print(f"Total Nodes: {len(simulation.nodes)}")
    print(f"Attacker Count: {attacker_count}")
    return simulation, attacker_count


def train(base_simulation, model_path: Path, seed: int):
    """Train the fraud model from extracted simulation features."""
    logging.info("Step 2/5: Training model...")
    features_df = extract_node_features(base_simulation)
    model_artifacts = _call_with_supported_kwargs(
        train_model,
        {
            "dataset": features_df,
            "dataframe": features_df,
            "model_path": model_path,
            "random_state": seed,
            "seed": seed,
        },
    )
    metrics = getattr(model_artifacts, "metrics", {})
    print("Model Metrics:")
    print(f"  Accuracy : {float(metrics.get('accuracy', 0.0)):.4f}")
    print(f"  Precision: {float(metrics.get('precision', 0.0)):.4f}")
    print(f"  Recall   : {float(metrics.get('recall', 0.0)):.4f}")
    print(f"  ROC-AUC  : {float(metrics.get('roc_auc', 0.0)):.4f}")
    return model_artifacts


def run_adaptive(base_simulation, model_artifacts, nodes: int, timesteps: int, seed: int):
    """Run adaptive simulation and compute detection-over-time metrics."""
    logging.info("Step 3/5: Running adaptive simulation...")
    if callable(run_adaptive_simulation):
        try:
            adaptive_simulation = _call_with_supported_kwargs(
                run_adaptive_simulation,
                {
                    "base_simulation": base_simulation,
                    "total_nodes": nodes,
                    "time_steps": timesteps,
                    "seed": seed + 7,
                    "model_for_adaptation": getattr(model_artifacts, "model", None),
                    "fraud_score_threshold": getattr(model_artifacts, "threshold", 0.5),
                    "feature_columns": FEATURE_COLUMNS,
                },
            )
        except (TypeError, ValueError) as adaptive_error:
            logging.warning(
                "run_adaptive_simulation fallback triggered: %s. Using build_network_simulation instead.",
                adaptive_error,
            )
            adaptive_simulation = build_network_simulation(
                total_nodes=nodes,
                time_steps=timesteps,
                seed=seed + 7,
                model_for_adaptation=getattr(model_artifacts, "model", None),
                fraud_score_threshold=getattr(model_artifacts, "threshold", 0.5),
                feature_columns=FEATURE_COLUMNS,
            )
    else:
        adaptive_simulation = build_network_simulation(
            total_nodes=nodes,
            time_steps=timesteps,
            seed=seed + 7,
            model_for_adaptation=getattr(model_artifacts, "model", None),
            fraud_score_threshold=getattr(model_artifacts, "threshold", 0.5),
            feature_columns=FEATURE_COLUMNS,
        )

    node_labels = {node.node_id: node.label for node in adaptive_simulation.nodes}
    detection_result = compute_detection_metrics(
        fraud_scores_over_time=adaptive_simulation.fraud_scores_over_time,
        node_labels=node_labels,
        threshold=float(getattr(model_artifacts, "threshold", 0.5)),
    )
    logging.info("Adaptive detection summary: %s", detection_result.summary)
    return adaptive_simulation, detection_result


def generate_predictions(adaptive_simulation, model_artifacts):
    """Generate fraud scores, binary predictions, and detected node ids."""
    logging.info("Step 4/5: Generating predictions...")
    features_df = extract_node_features(adaptive_simulation)
    scores = model_artifacts.model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1]
    threshold = float(getattr(model_artifacts, "threshold", 0.5))
    predicted = (scores >= threshold).astype(int)
    detected_node_ids = features_df.loc[predicted == 1, "node_id"].tolist()
    return features_df, scores, predicted, detected_node_ids, threshold


def compute_business_metrics(features_df: pd.DataFrame, predicted, threshold: float, adaptive_simulation):
    """Compute attacker detection %, false-positive %, and fraud-loss reduction %."""
    logging.info("Step 5/5: Computing business metrics...")
    y_true = features_df["label"].astype(int)
    total_attackers = int((y_true == 1).sum())
    total_honest = int((y_true == 0).sum())
    true_positives = int(((y_true == 1) & (predicted == 1)).sum())
    false_positives = int(((y_true == 0) & (predicted == 1)).sum())
    attacker_detection_pct = 100.0 * true_positives / max(total_attackers, 1)
    false_positive_rate_pct = 100.0 * false_positives / max(total_honest, 1)

    fraud_loss_reduction_pct = 0.0
    economic_result = None
    if simulate_economic_rewards is not None:
        node_labels = {node.node_id: node.label for node in adaptive_simulation.nodes}
        node_regions = {node.node_id: node.region for node in adaptive_simulation.nodes}
        connectivity_over_time = []
        for snapshot in adaptive_simulation.time_steps:
            graph = snapshot.peer_graph
            degree_map = {int(node_id): int(graph.degree(node_id)) for node_id in graph.nodes}
            max_degree = max(degree_map.values(), default=1)
            connectivity_over_time.append(
                {node_id: float(degree / max(max_degree, 1)) for node_id, degree in degree_map.items()}
            )
        economic_result = simulate_economic_rewards(
            fraud_scores_over_time=adaptive_simulation.fraud_scores_over_time,
            node_labels=node_labels,
            threshold=threshold,
            node_regions=node_regions,
            connectivity_over_time=connectivity_over_time,
        )
        summary = economic_result.summary
        reduced = float(summary.get("fraud_reduction_after_detection", 0.0))
        remaining = float(summary.get("total_fraud_profit", 0.0))
        fraud_loss_reduction_pct = 100.0 * reduced / max(reduced + remaining, 1e-9)

    return {
        "attackers": total_attackers,
        "attacker_detection_pct": attacker_detection_pct,
        "false_positive_rate_pct": false_positive_rate_pct,
        "fraud_loss_reduction_pct": fraud_loss_reduction_pct,
        "economic_result": economic_result,
    }


def _save_plot(fig, ax, title: str, output_path: Path):
    if ax is not None:
        ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def visualize(
    output_dir: Path,
    adaptive_simulation,
    features_df: pd.DataFrame,
    scores,
    detected_node_ids,
    detection_result,
    economic_result,
):
    """Generate and save all required demo visualizations."""
    logging.info("Generating visualizations in %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_network(adaptive_simulation, detected_fraud_nodes=detected_node_ids, output_path=None)
    _save_plot(fig, ax, "Decentralized Network Simulation", output_dir / "network.png")

    fig, ax = plot_fraud_score_distribution(
        fraud_scores=scores,
        labels=features_df["label"].to_numpy(dtype=int),
        output_path=None,
    )
    _save_plot(fig, ax, "Fraud Score Distribution", output_dir / "fraud_distribution.png")

    fig, ax = plot_detection_over_time(
        false_positives_over_time=detection_result.false_positives_over_time,
        output_path=None,
    )
    _save_plot(fig, ax, "Detection Over Time", output_dir / "detection_time.png")

    if economic_result is not None:
        fig, ax = plot_reward_distribution(per_node_rewards=economic_result.per_node_rewards, output_path=None)
        _save_plot(fig, ax, "Reward Distribution: Honest vs Attackers", output_dir / "rewards.png")

    fig, ax = plot_roc_curve(
        y_true=features_df["label"].to_numpy(dtype=int),
        y_scores=scores,
        output_path=None,
    )
    _save_plot(fig, ax, "ROC Curve", output_dir / "roc_curve.png")


def main():
    """Execute the complete end-to-end demo pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    model_path = output_dir / "model.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_simulation, attacker_count = run_simulation(args.nodes, args.timesteps, args.seed)
        model_artifacts = train(base_simulation, model_path=model_path, seed=args.seed)
        adaptive_simulation, detection_result = run_adaptive(
            base_simulation=base_simulation,
            model_artifacts=model_artifacts,
            nodes=args.nodes,
            timesteps=args.timesteps,
            seed=args.seed,
        )
        features_df, scores, predicted, detected_node_ids, threshold = generate_predictions(
            adaptive_simulation=adaptive_simulation,
            model_artifacts=model_artifacts,
        )
        business_metrics = compute_business_metrics(
            features_df=features_df,
            predicted=predicted,
            threshold=threshold,
            adaptive_simulation=adaptive_simulation,
        )
        visualize(
            output_dir=output_dir,
            adaptive_simulation=adaptive_simulation,
            features_df=features_df,
            scores=scores,
            detected_node_ids=detected_node_ids,
            detection_result=detection_result,
            economic_result=business_metrics["economic_result"],
        )
    except Exception as exc:
        logging.exception("Demo pipeline failed. Check logs above for stage details.")
        raise SystemExit(1) from exc

    print("\n## DEMO RESULTS\n")
    print(f"Total Nodes: {args.nodes}")
    print(f"Total Attackers: {attacker_count}")
    print(f"Detected Attackers: {business_metrics['attacker_detection_pct']:.2f}%")
    print(f"False Positives: {business_metrics['false_positive_rate_pct']:.2f}%")
    print(f"Fraud Loss Reduced: {business_metrics['fraud_loss_reduction_pct']:.2f}%")
    print("-----------------------")


if __name__ == "__main__":
    main()
