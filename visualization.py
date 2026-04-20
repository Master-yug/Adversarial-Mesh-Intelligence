from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

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
    output_path: str | None = None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    honest_scores = fraud_scores[labels == 0]
    attacker_scores = fraud_scores[labels == 1]
    ax.hist(honest_scores, bins=24, alpha=0.65, label="Honest", color="#22c55e", density=True)
    ax.hist(attacker_scores, bins=24, alpha=0.65, label="Attacker", color="#ef4444", density=True)
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
