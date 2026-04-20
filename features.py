from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from constants import AGGREGATION_SUFFIXES, BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, FEATURE_MEANINGS
from geo import haversine_km
from simulation import SimulationResult

Coordinate = Tuple[float, float]
# Heuristic trust-score parameters tuned to emphasize stable reciprocal neighborhoods
# while still penalizing volatile neighbor latency behavior.
TRUST_DISPERSION_SCALE = 150.0
TRUST_RECIPROCITY_WEIGHT = 0.65
TRUST_STABILITY_WEIGHT = 0.35


@dataclass(frozen=True)
class _FallbackSnapshot:
    timestep: int
    peer_graph: nx.DiGraph
    latency_matrix: np.ndarray


def _safe_array(values: Iterable[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr[np.isfinite(arr)]


def _robust_median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmedian(values))


def _robust_var(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.nanvar(values))


def _skewness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    mu = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std < 1e-9:
        return 0.0
    centered = (values - mu) / std
    return float(np.nanmean(centered**3))


def _kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    mu = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std < 1e-9:
        return 0.0
    centered = (values - mu) / std
    return float(np.nanmean(centered**4) - 3.0)


def _infer_location_from_peers(
    node_id: int,
    peers: Iterable[int],
    latency_matrix: np.ndarray,
    claimed_locations: Dict[int, Coordinate],
) -> Coordinate:
    peer_list = [p for p in peers if p in claimed_locations and p != node_id]
    if not peer_list:
        return claimed_locations[node_id]

    peer_coords = np.array([claimed_locations[p] for p in peer_list], dtype=float)
    peer_latencies = np.array([latency_matrix[node_id, p] for p in peer_list], dtype=float)
    valid = np.isfinite(peer_latencies)
    if not np.any(valid):
        return claimed_locations[node_id]

    peer_coords = peer_coords[valid]
    peer_latencies = peer_latencies[valid]
    weights = 1.0 / np.maximum(peer_latencies, 1.0)
    weights = weights / np.sum(weights)
    inferred = np.average(peer_coords, axis=0, weights=weights)
    return float(inferred[0]), float(inferred[1])


def _peer_geographic_diversity(peer_coords: np.ndarray) -> float:
    if peer_coords.shape[0] < 2:
        return 0.0
    center = np.nanmean(peer_coords, axis=0)
    dists = np.array([haversine_km((c[0], c[1]), (center[0], center[1])) for c in peer_coords], dtype=float)
    return float(np.nanmedian(dists))


def _expected_latency_from_claimed_geo(source_claimed: Coordinate, target_claimed: Coordinate) -> float:
    distance = haversine_km(source_claimed, target_claimed)
    base = 5.0 + 0.0105 * distance + 0.0002 * (distance**1.23)
    return float(max(1.0, base))


def _latency_inconsistency(
    node_id: int,
    out_neighbors: Iterable[int],
    latency_matrix: np.ndarray,
    claimed_locations: Dict[int, Coordinate],
) -> float:
    residuals = []
    node_claimed = claimed_locations[node_id]
    for peer in out_neighbors:
        observed = latency_matrix[node_id, peer]
        if not np.isfinite(observed):
            continue
        expected = _expected_latency_from_claimed_geo(node_claimed, claimed_locations[peer])
        residuals.append(abs(observed - expected) / max(expected, 1.0))
    return _robust_median(_safe_array(residuals))


def _neighbor_trust_score(
    node_id: int,
    graph: nx.DiGraph,
    latency_matrix: np.ndarray,
    all_neighbor_ids: Iterable[int],
) -> float:
    peers = list(all_neighbor_ids)
    if not peers:
        return 0.0

    reciprocity_values = []
    peer_latency_dispersion = []
    for peer in peers:
        reciprocity = nx.reciprocity(graph, peer)
        reciprocity_values.append(0.0 if reciprocity is None else float(reciprocity))
        lat = _safe_array(latency_matrix[peer, :])
        if lat.size > 1:
            peer_latency_dispersion.append(float(np.nanstd(lat)))

    rec = _robust_median(_safe_array(reciprocity_values))
    dispersion = _robust_median(_safe_array(peer_latency_dispersion))
    dispersion_penalty = min(1.0, dispersion / TRUST_DISPERSION_SCALE)
    return float(
        np.clip(
            TRUST_RECIPROCITY_WEIGHT * rec + TRUST_STABILITY_WEIGHT * (1.0 - dispersion_penalty),
            0.0,
            1.0,
        )
    )


def _edge_asymmetry(node_id: int, neighbors: Iterable[int], latency_matrix: np.ndarray) -> float:
    asym = []
    for peer in neighbors:
        a_to_b = latency_matrix[node_id, peer]
        b_to_a = latency_matrix[peer, node_id]
        if not np.isfinite(a_to_b) or not np.isfinite(b_to_a):
            continue
        asym.append(abs(a_to_b - b_to_a) / max((a_to_b + b_to_a) / 2.0, 1.0))
    return _robust_median(_safe_array(asym))


def _compute_temporal_anomaly_features(node_history: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal anomaly features from a node's ordered history.

    Expects columns: timestep, avg_latency_to_peers, unique_peers, clustering_coefficient,
    reciprocity_score. Adds max_latency_spike, latency_trend_slope, behavior_volatility,
    sudden_change_score, and burst_activity_score.
    """
    history = node_history.sort_values("timestep").copy()
    avg_latency = history["avg_latency_to_peers"].to_numpy(dtype=float)
    unique_peers = history["unique_peers"].to_numpy(dtype=float)
    clustering = history["clustering_coefficient"].to_numpy(dtype=float)
    reciprocity = history["reciprocity_score"].to_numpy(dtype=float)

    if avg_latency.size == 0:
        history["max_latency_spike"] = 0.0
        history["latency_trend_slope"] = 0.0
        history["behavior_volatility"] = 0.0
        history["sudden_change_score"] = 0.0
        history["burst_activity_score"] = 0.0
        return history

    running_max_spike = []
    running_slope = []
    running_volatility = []
    running_sudden_change = []
    running_burst_score = []

    for i in range(avg_latency.size):
        lat_slice = avg_latency[: i + 1]
        peer_slice = unique_peers[: i + 1]
        cluster_slice = clustering[: i + 1]
        reciprocity_slice = reciprocity[: i + 1]

        if lat_slice.size >= 2:
            latency_diff = np.diff(lat_slice)
            max_spike = float(np.max(np.abs(latency_diff)))
            sudden_change = float(np.mean(np.abs(latency_diff)))
        else:
            latency_diff = np.array([], dtype=float)
            max_spike = 0.0
            sudden_change = 0.0

        if lat_slice.size >= 3:
            x = np.arange(lat_slice.size, dtype=float)
            slope = float(np.polyfit(x, lat_slice, deg=1)[0])
        else:
            slope = 0.0

        stacked = np.column_stack([lat_slice, peer_slice, cluster_slice, reciprocity_slice])
        behavior_volatility = float(np.nanmean(np.nanstd(stacked, axis=0, ddof=0)))

        if lat_slice.size >= 2:
            burst_threshold = float(np.nanmean(np.abs(latency_diff)) + np.nanstd(np.abs(latency_diff)))
            burst_count = int(np.sum(np.abs(latency_diff) > burst_threshold))
            burst_score = float(burst_count / max(latency_diff.size, 1))
        else:
            burst_score = 0.0

        running_max_spike.append(max_spike)
        running_slope.append(slope)
        running_volatility.append(behavior_volatility)
        running_sudden_change.append(sudden_change)
        running_burst_score.append(burst_score)

    history["max_latency_spike"] = running_max_spike
    history["latency_trend_slope"] = running_slope
    history["behavior_volatility"] = running_volatility
    history["sudden_change_score"] = running_sudden_change
    history["burst_activity_score"] = running_burst_score
    return history


def _extract_features_from_graph(
    graph: nx.DiGraph,
    latency_matrix: np.ndarray,
    nodes_meta: Dict[int, Dict[str, object]],
    timestep: int,
) -> pd.DataFrame:
    claimed_locations = {
        node_id: tuple(meta["claimed_location"])  # type: ignore[misc]
        for node_id, meta in nodes_meta.items()
    }
    undirected_graph = graph.to_undirected()
    clustering = nx.clustering(undirected_graph)

    rows = []
    for node_id in graph.nodes:
        out_neighbors = set(graph.successors(node_id))
        in_neighbors = set(graph.predecessors(node_id))
        all_neighbors = out_neighbors | in_neighbors

        peer_latencies = _safe_array([latency_matrix[node_id, p] for p in out_neighbors])
        if peer_latencies.size == 0:
            peer_latencies = np.array([0.0], dtype=float)

        inferred_location = _infer_location_from_peers(
            node_id=node_id,
            peers=all_neighbors,
            latency_matrix=latency_matrix,
            claimed_locations=claimed_locations,
        )
        mismatch_km = haversine_km(claimed_locations[node_id], inferred_location)

        peer_coords = np.array([claimed_locations[p] for p in all_neighbors], dtype=float) if all_neighbors else np.zeros((0, 2))

        reciprocity = nx.reciprocity(graph, node_id)
        reciprocity_score = 0.0 if reciprocity is None else float(reciprocity)

        feature_row = {
            "node_id": int(node_id),
            "timestep": int(timestep),
            "rtt_variance": _robust_var(peer_latencies),
            "avg_latency_to_peers": _robust_median(peer_latencies),
            "claimed_inferred_distance_mismatch": float(mismatch_km),
            "unique_peers": float(len(all_neighbors)),
            "clustering_coefficient": float(clustering.get(node_id, 0.0)),
            "reciprocity_score": reciprocity_score,
            "latency_skewness": _skewness(peer_latencies),
            "latency_kurtosis": _kurtosis(peer_latencies),
            "peer_geographic_diversity": _peer_geographic_diversity(peer_coords),
            "latency_inconsistency_score": _latency_inconsistency(
                node_id=node_id,
                out_neighbors=out_neighbors,
                latency_matrix=latency_matrix,
                claimed_locations=claimed_locations,
            ),
            "neighbor_trust_score": _neighbor_trust_score(
                node_id=node_id,
                graph=graph,
                latency_matrix=latency_matrix,
                all_neighbor_ids=all_neighbors,
            ),
            "edge_asymmetry": _edge_asymmetry(node_id=node_id, neighbors=all_neighbors, latency_matrix=latency_matrix),
        }

        label_name = str(nodes_meta[node_id]["label"])
        feature_row["label_name"] = label_name
        feature_row["label"] = 0 if label_name == "honest" else 1
        rows.append(feature_row)

    return pd.DataFrame(rows).sort_values(["timestep", "node_id"]).reset_index(drop=True)


def extract_snapshot_features(
    graph: nx.DiGraph,
    latency_matrix: np.ndarray,
    nodes_meta: Dict[int, Dict[str, object]],
    timestep: int,
) -> pd.DataFrame:
    return _extract_features_from_graph(
        graph=graph,
        latency_matrix=latency_matrix,
        nodes_meta=nodes_meta,
        timestep=timestep,
    )


def extract_temporal_node_features(simulation: SimulationResult) -> pd.DataFrame:
    nodes_meta = {
        n.node_id: {
            "label": n.label,
            "claimed_location": (n.claimed_lat, n.claimed_lon),
        }
        for n in simulation.nodes
    }
    snapshots = simulation.time_steps or []
    if not snapshots:
        snapshots = [
            _FallbackSnapshot(
                timestep=0,
                peer_graph=simulation.peer_graph,
                latency_matrix=simulation.latency_matrix,
            )
        ]

    temporal_frames = [
        _extract_features_from_graph(
            graph=snapshot.peer_graph,
            latency_matrix=snapshot.latency_matrix,
            nodes_meta=nodes_meta,
            timestep=snapshot.timestep,
        )
        for snapshot in snapshots
    ]
    temporal_df = pd.concat(temporal_frames, ignore_index=True).sort_values(["node_id", "timestep"]).reset_index(drop=True)
    enriched_frames = []
    for _, node_history in temporal_df.groupby("node_id", sort=False):
        enriched_frames.append(_compute_temporal_anomaly_features(node_history))
    enriched = pd.concat(enriched_frames, ignore_index=True).sort_values(["timestep", "node_id"]).reset_index(drop=True)
    return enriched


def build_model_features_from_temporal_frame(temporal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in temporal_df.iterrows():
        out = {
            "node_id": int(row["node_id"]),
            "timestep": int(row["timestep"]),
            "label": int(row["label"]),
            "label_name": str(row["label_name"]),
        }
        for base_feature in BASE_FEATURE_COLUMNS:
            value = float(row.get(base_feature, 0.0))
            out[f"{base_feature}_mean"] = value
            out[f"{base_feature}_std"] = 0.0
            out[f"{base_feature}_last"] = value
        rows.append(out)
    model_df = pd.DataFrame(rows)
    return model_df[["node_id", "timestep", *FEATURE_COLUMNS, "label", "label_name"]].sort_values(
        ["timestep", "node_id"]
    ).reset_index(drop=True)


def _aggregate_temporal_features(temporal_df: pd.DataFrame) -> pd.DataFrame:
    aggregate_frames = []
    for base_feature in BASE_FEATURE_COLUMNS:
        grouped = temporal_df.groupby("node_id")[base_feature]
        aggregate_frames.append(grouped.mean().rename(f"{base_feature}_mean"))
        aggregate_frames.append(grouped.std(ddof=0).fillna(0.0).rename(f"{base_feature}_std"))
        aggregate_frames.append(grouped.last().rename(f"{base_feature}_last"))

    aggregated = pd.concat(aggregate_frames, axis=1).reset_index()
    label_frame = temporal_df.groupby("node_id")[["label", "label_name"]].last().reset_index()
    merged = aggregated.merge(label_frame, on="node_id", how="left")

    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
    merged = merged[["node_id", *FEATURE_COLUMNS, "label", "label_name"]]
    return merged.sort_values("node_id").reset_index(drop=True)


def extract_node_features(simulation: SimulationResult) -> pd.DataFrame:
    temporal_df = extract_temporal_node_features(simulation)
    aggregated_df = _aggregate_temporal_features(temporal_df)
    numeric_cols = [c for c in aggregated_df.columns if c not in {"node_id", "label_name"}]
    aggregated_df[numeric_cols] = aggregated_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return aggregated_df


def feature_meanings_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [{"feature": feature, "meaning": meaning} for feature, meaning in FEATURE_MEANINGS.items()]
    ).sort_values("feature")


__all__ = [
    "extract_node_features",
    "extract_snapshot_features",
    "extract_temporal_node_features",
    "build_model_features_from_temporal_frame",
    "feature_meanings_dataframe",
    "FEATURE_COLUMNS",
    "AGGREGATION_SUFFIXES",
]
