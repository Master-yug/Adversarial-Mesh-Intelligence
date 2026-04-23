from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from simulation import SimulationResult


def plot_network(
    simulation: SimulationResult,
    detected_fraud_nodes: Iterable[int] | None = None,
    output_path: str | None = None,
):
    graph = simulation.peer_graph
    detected = set(detected_fraud_nodes or [])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(graph, seed=42, k=0.22, iterations=150)

    labels = nx.get_node_attributes(graph, "label")
    honest_nodes = [n for n, l in labels.items() if l == "honest"]
    naive_attacker_nodes = [n for n, l in labels.items() if l == "naive_attacker"]
    smart_attacker_nodes = [n for n, l in labels.items() if l == "smart_attacker"]

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        alpha=0.20,
        arrows=False,
        width=0.5,
        edge_color="#94a3b8",
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=honest_nodes,
        node_color="#22c55e",
        node_size=70,
        alpha=0.9,
        ax=ax,
        label="Honest",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=naive_attacker_nodes,
        node_color="#ef4444",
        node_size=80,
        alpha=0.95,
        ax=ax,
        label="Naive attacker",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=smart_attacker_nodes,
        node_color="#f97316",
        node_size=84,
        alpha=0.95,
        ax=ax,
        label="Smart attacker",
    )

    if detected:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=list(detected),
            node_size=210,
            node_color="none",
            edgecolors="#facc15",
            linewidths=2.3,
            ax=ax,
            label="Detected fraud",
        )

    ax.set_title("Decentralized Network Simulation", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig, ax


def plot_fraud_score_distribution(
    fraud_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    output_path: str | None = None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    honest_scores = fraud_scores[labels == 0]
    attacker_scores = fraud_scores[labels == 1]
    ax.hist(honest_scores, bins=24, alpha=0.65, label="Honest", color="#22c55e", density=True)
    ax.hist(attacker_scores, bins=24, alpha=0.65, label="Attacker", color="#ef4444", density=True)
    if honest_scores.size > 1 and attacker_scores.size > 1:
        bins = np.linspace(0.0, 1.0, 50)
        honest_hist, _ = np.histogram(honest_scores, bins=bins, density=True)
        attacker_hist, _ = np.histogram(attacker_scores, bins=bins, density=True)
        overlap = float(np.sum(np.minimum(honest_hist, attacker_hist)) * (bins[1] - bins[0]))
        ax.text(0.02, 0.95, f"overlap={overlap:.2f}", transform=ax.transAxes, fontsize=10, va="top")
    if threshold is not None:
        ax.axvline(float(threshold), linestyle="--", color="#111827", linewidth=1.6, label="Decision threshold")
    ax.set_title("Fraud Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fraud score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center", ncols=2, frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label="Model ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.5, label="Random baseline")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str | None = None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(recall, precision, color="#0ea5e9", linewidth=2.5, label="Model PR")
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, linestyle="--", color="#64748b", linewidth=1.5, label="Class balance baseline")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_calibration_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str | None = None):
    frac_pos, mean_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy="quantile")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2.2, color="#8b5cf6", label="Model calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.5, label="Perfect calibration")
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_performance_vs_noise(noise_df: pd.DataFrame, output_path: str | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(noise_df["noise_level"], noise_df["roc_auc"], marker="o", linewidth=2.2, color="#2563eb", label="ROC-AUC")
    ax.plot(noise_df["noise_level"], noise_df["false_positive_rate"], marker="s", linewidth=2.0, color="#dc2626", label="FPR")
    ax.set_title("Performance vs Noise Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_detection_over_time(false_positives_over_time: pd.DataFrame, output_path: str | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        false_positives_over_time["timestep"],
        false_positives_over_time["false_positives"],
        marker="o",
        linewidth=2.2,
        color="#f59e0b",
        label="False positives",
    )
    ax.set_title("Detection Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Flagged nodes")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_reward_distribution(per_node_rewards: pd.DataFrame, output_path: str | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    honest = per_node_rewards.loc[per_node_rewards["label_name"] == "honest", "total_reward"].to_numpy()
    attacker = per_node_rewards.loc[per_node_rewards["label_name"] != "honest", "total_reward"].to_numpy()
    ax.hist(honest, bins=22, alpha=0.65, label="Honest", color="#22c55e", density=True)
    ax.hist(attacker, bins=22, alpha=0.65, label="Attackers", color="#dc2626", density=True)
    ax.set_title("Reward Distribution: Honest vs Attackers", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total reward")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center", ncols=2, frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_strategy_distribution_over_time(strategy_distribution_history: list[dict[str, float]], output_path: str | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    if strategy_distribution_history:
        strategy_names = sorted({k for row in strategy_distribution_history for k in row.keys()})
        x = np.arange(len(strategy_distribution_history))
        for strategy_name in strategy_names:
            y = [float(row.get(strategy_name, 0.0)) for row in strategy_distribution_history]
            ax.plot(x, y, linewidth=2.0, marker="o", label=strategy_name)
    ax.set_title("Attacker Strategy Distribution Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Strategy probability")
    ax.set_ylim(0.0, 1.0)
    if strategy_distribution_history:
        ax.legend(loc="upper right", ncols=2, frameon=True)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_defender_threshold_over_time(threshold_history: list[float], output_path: str | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(threshold_history))
    y = np.array(threshold_history, dtype=float) if threshold_history else np.array([], dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.2, color="#2563eb")
    ax.set_title("Defender Threshold Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Threshold")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_cost_vs_time(cost_history: list[float], output_path: str | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cost_history))
    y = np.array(cost_history, dtype=float) if cost_history else np.array([], dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.2, color="#dc2626")
    ax.set_title("System Cost vs Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total system cost")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_fraud_leakage_vs_time(fraud_leakage_history: list[float], output_path: str | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fraud_leakage_history))
    y = np.array(fraud_leakage_history, dtype=float) if fraud_leakage_history else np.array([], dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.2, color="#ea580c")
    ax.set_title("Fraud Leakage vs Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraud leakage (%)")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, ax


def plot_equilibrium_detection(
    equilibrium_detected_history: list[float],
    strategy_delta_history: list[float],
    cost_delta_ratio_history: list[float],
    output_path: str | None = None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(equilibrium_detected_history))
    eq = np.array(equilibrium_detected_history, dtype=float) if equilibrium_detected_history else np.array([], dtype=float)
    sd = np.array(strategy_delta_history, dtype=float) if strategy_delta_history else np.array([], dtype=float)
    cd = np.array(cost_delta_ratio_history, dtype=float) if cost_delta_ratio_history else np.array([], dtype=float)

    ax1.plot(x, eq, marker="o", linewidth=2.2, color="#16a34a", label="Equilibrium reached")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Equilibrium indicator", color="#166534")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(x, sd, marker="s", linewidth=1.8, color="#2563eb", label="Strategy delta")
    ax2.plot(x, cd, marker="^", linewidth=1.8, color="#dc2626", label="Cost delta ratio")
    ax2.set_ylabel("Convergence deltas")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    ax1.set_title("Equilibrium Detection Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=220)
    return fig, (ax1, ax2)
