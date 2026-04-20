from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from geo import haversine_km

Coordinate = Tuple[float, float]


CITY_METADATA: Dict[str, Dict[str, object]] = {
    "new_york": {"coord": (40.7128, -74.0060), "continent": "north_america"},
    "san_francisco": {"coord": (37.7749, -122.4194), "continent": "north_america"},
    "toronto": {"coord": (43.6532, -79.3832), "continent": "north_america"},
    "mexico_city": {"coord": (19.4326, -99.1332), "continent": "north_america"},
    "london": {"coord": (51.5074, -0.1278), "continent": "europe"},
    "paris": {"coord": (48.8566, 2.3522), "continent": "europe"},
    "berlin": {"coord": (52.5200, 13.4050), "continent": "europe"},
    "madrid": {"coord": (40.4168, -3.7038), "continent": "europe"},
    "tokyo": {"coord": (35.6762, 139.6503), "continent": "asia"},
    "singapore": {"coord": (1.3521, 103.8198), "continent": "asia"},
    "dubai": {"coord": (25.2048, 55.2708), "continent": "asia"},
    "mumbai": {"coord": (19.0760, 72.8777), "continent": "asia"},
    "sydney": {"coord": (-33.8688, 151.2093), "continent": "oceania"},
    "melbourne": {"coord": (-37.8136, 144.9631), "continent": "oceania"},
    "sao_paulo": {"coord": (-23.5558, -46.6396), "continent": "south_america"},
    "johannesburg": {"coord": (-26.2041, 28.0473), "continent": "africa"},
}

SMART_ATTACKER_BASE_ADAPTATION = 0.08
SMART_ATTACKER_ADAPTATION_GROWTH = 0.10
SMART_ATTACKER_STRATEGIES = ("low_and_slow", "burst_attack", "camouflage")
RATIO_SUM_TOLERANCE = 1e-6
DEFAULT_TARGET_FEATURE_PRIORITY = (
    "clustering_coefficient",
    "rtt_variance",
    "peer_geographic_diversity",
    "claimed_inferred_distance_mismatch",
    "unique_peers",
)
FEATURE_TARGET_ACTIONS = {
    "clustering_coefficient": "reduce_clustering",
    "rtt_variance": "add_noise",
    "latency_inconsistency_score": "add_noise",
    "latency_skewness": "add_noise",
    "latency_kurtosis": "add_noise",
    "peer_geographic_diversity": "increase_connections",
    "unique_peers": "increase_connections",
    "claimed_inferred_distance_mismatch": "increase_connections",
}
STEALTH_ACTION_COSTS = {
    "reduce_clustering": 0.28,
    "add_noise": 0.24,
    "increase_connections": 0.26,
}
STEALTH_REGEN_RATE = 0.06
REGIONAL_ATTACK_INJECTION_PROB = 0.18
SYBIL_SWARM_INJECTION_PROB = 0.16
CASCADING_FAILURE_INJECTION_PROB = 0.14
NAIVE_ATTACKER_TARGET_REGION_PROB = 0.70
SMART_ATTACKER_TARGET_REGION_PROB = 0.75
MIN_SYBIL_SWARM_SIZE = 3
SYBIL_SWARM_FRACTION = 0.45
MIN_CASCADING_SPREAD = 1
CASCADING_SPREAD_FRACTION = 0.05
MIN_CORRUPTED_LATENCY = 1.0
CORRUPTED_LATENCY_MEAN = 10.0
CORRUPTED_LATENCY_STD = 8.0
SYBIL_LATENCY_REDUCTION = 0.45
MIN_SYBIL_LATENCY = 1.0
SYBIL_LATENCY_NOISE_STD = 3.0
REGIONAL_ATTACK_LATENCY_FACTOR = 0.85
MAX_SYBIL_ATTACKER_RATIO = 0.98
SYBIL_ATTACKER_RATIO_BOOST = 0.25


@dataclass(frozen=True)
class NodeRecord:
    node_id: int
    label: str  # honest / naive_attacker / smart_attacker
    real_lat: float
    real_lon: float
    claimed_lat: float
    claimed_lon: float
    continent: str
    region: str
    anomalous_honest: bool
    unstable_connection: bool
    attacker_strategy: str | None = None


@dataclass(frozen=True)
class TimeStepSnapshot:
    timestep: int
    latency_matrix: np.ndarray
    peer_graph: nx.DiGraph
    fraud_scores: Dict[int, float]


@dataclass(frozen=True)
class AttackScenarioEvent:
    timestep: int
    scenario_name: str
    impacted_node_count: int


@dataclass(frozen=True)
class SimulationResult:
    nodes: List[NodeRecord]
    latency_matrix: np.ndarray
    peer_graph: nx.DiGraph
    time_steps: List[TimeStepSnapshot]
    fraud_scores_over_time: List[Dict[int, float]]
    scenario_events: List[AttackScenarioEvent]
    scenario_metrics: Dict[str, float]


def _city_meta_or_default(city: str) -> Dict[str, object]:
    return CITY_METADATA.get(city, CITY_METADATA["london"])


def _continent_penalty(continent_a: str, continent_b: str, rng: np.random.Generator) -> float:
    if continent_a == continent_b:
        return float(rng.uniform(3.0, 25.0))
    return float(rng.uniform(50.0, 150.0))


def _sample_honest_nodes(
    node_ids: Sequence[int],
    rng: np.random.Generator,
    anomaly_rate: float,
    unstable_rate: float,
) -> List[NodeRecord]:
    city_names = list(CITY_METADATA.keys())
    rows: List[NodeRecord] = []
    for node_id in node_ids:
        city = city_names[int(rng.integers(0, len(city_names)))]
        city_meta = _city_meta_or_default(city)
        base_lat, base_lon = city_meta["coord"]  # type: ignore[index]
        lat = float(base_lat + rng.normal(0.0, 1.9))
        lon = float(base_lon + rng.normal(0.0, 2.2))
        is_anomalous = bool(rng.random() < anomaly_rate)
        is_unstable = bool(rng.random() < unstable_rate)
        rows.append(
            NodeRecord(
                node_id=int(node_id),
                label="honest",
                real_lat=lat,
                real_lon=lon,
                claimed_lat=lat,
                claimed_lon=lon,
                continent=str(city_meta["continent"]),
                region=city,
                anomalous_honest=is_anomalous,
                unstable_connection=is_unstable,
                attacker_strategy=None,
            )
        )
    return rows


def _sample_naive_attackers(
    node_ids: Sequence[int],
    cluster_center: Coordinate,
    cluster_continent: str,
    rng: np.random.Generator,
    target_region: str | None = None,
) -> List[NodeRecord]:
    city_names = list(CITY_METADATA.keys())
    rows: List[NodeRecord] = []
    for node_id in node_ids:
        real_lat = float(rng.normal(cluster_center[0], 0.22))
        real_lon = float(rng.normal(cluster_center[1], 0.28))
        fake_city = city_names[int(rng.integers(0, len(city_names)))]
        if target_region and rng.random() < NAIVE_ATTACKER_TARGET_REGION_PROB:
            fake_city = target_region
        fake_city_meta = _city_meta_or_default(fake_city)
        fake_lat, fake_lon = fake_city_meta["coord"]  # type: ignore[index]
        rows.append(
            NodeRecord(
                node_id=int(node_id),
                label="naive_attacker",
                real_lat=real_lat,
                real_lon=real_lon,
                claimed_lat=float(fake_lat),
                claimed_lon=float(fake_lon),
                continent=cluster_continent,
                region=fake_city,
                anomalous_honest=False,
                unstable_connection=False,
                attacker_strategy=None,
            )
        )
    return rows


def _sample_smart_attacker_strategies(
    node_ids: Sequence[int],
    rng: np.random.Generator,
    strategy_mix: Dict[str, float] | None = None,
) -> Dict[int, str]:
    mix = strategy_mix or {"low_and_slow": 0.4, "burst_attack": 0.3, "camouflage": 0.3}
    probs = np.array([float(mix.get(strategy, 0.0)) for strategy in SMART_ATTACKER_STRATEGIES], dtype=float)
    if probs.sum() <= 0:
        probs = np.array([0.4, 0.3, 0.3], dtype=float)
    probs = probs / probs.sum()
    strategies = rng.choice(SMART_ATTACKER_STRATEGIES, size=len(node_ids), replace=True, p=probs)
    return {int(node_id): str(strategy) for node_id, strategy in zip(node_ids, strategies)}


def _sample_smart_attackers(
    node_ids: Sequence[int],
    strategies: Dict[int, str],
    rng: np.random.Generator,
    target_region: str | None = None,
) -> List[NodeRecord]:
    city_names = list(CITY_METADATA.keys())
    rows: List[NodeRecord] = []
    for node_id in node_ids:
        real_city = city_names[int(rng.integers(0, len(city_names)))]
        real_city_meta = _city_meta_or_default(real_city)
        base_lat, base_lon = real_city_meta["coord"]  # type: ignore[index]
        real_lat = float(base_lat + rng.normal(0.0, 1.1))
        real_lon = float(base_lon + rng.normal(0.0, 1.3))

        strategy = strategies[int(node_id)]
        fake_probability = 0.20 if strategy == "low_and_slow" else 0.35
        if target_region and rng.random() < SMART_ATTACKER_TARGET_REGION_PROB:
            fake_city = target_region
            fake_city_meta = _city_meta_or_default(fake_city)
            fake_lat, fake_lon = fake_city_meta["coord"]  # type: ignore[index]
            claimed_lat = float(fake_lat + rng.normal(0.0, 0.5))
            claimed_lon = float(fake_lon + rng.normal(0.0, 0.5))
            region = fake_city
        elif rng.random() < fake_probability:
            fake_city = city_names[int(rng.integers(0, len(city_names)))]
            fake_city_meta = _city_meta_or_default(fake_city)
            fake_lat, fake_lon = fake_city_meta["coord"]  # type: ignore[index]
            claimed_lat = float(fake_lat + rng.normal(0.0, 0.7))
            claimed_lon = float(fake_lon + rng.normal(0.0, 0.7))
            region = fake_city
        else:
            claimed_lat = float(real_lat + rng.normal(0.0, 0.4))
            claimed_lon = float(real_lon + rng.normal(0.0, 0.4))
            region = real_city

        rows.append(
            NodeRecord(
                node_id=int(node_id),
                label="smart_attacker",
                real_lat=real_lat,
                real_lon=real_lon,
                claimed_lat=claimed_lat,
                claimed_lon=claimed_lon,
                continent=str(real_city_meta["continent"]),
                region=region,
                anomalous_honest=False,
                unstable_connection=False,
                attacker_strategy=strategy,
            )
        )
    return rows


def _initialize_smart_states(
    nodes: Sequence[NodeRecord],
    total_timesteps: int,
    rng: np.random.Generator,
    feature_priority: Sequence[str],
) -> Dict[int, Dict[str, object]]:
    """Create per-smart-attacker adaptive state with stealth budgets and feature targeting priority."""
    states: Dict[int, Dict[str, object]] = {}
    for node in nodes:
        if node.label != "smart_attacker":
            continue
        strategy = node.attacker_strategy or "low_and_slow"
        burst_start = int(
            rng.integers(
                max(1, total_timesteps // 3),
                max(2, (2 * total_timesteps) // 3 + 1),
            )
        )
        states[node.node_id] = {
            "strategy": strategy,
            "adaptation_strength": SMART_ATTACKER_BASE_ADAPTATION,
            "clustering_bias": 0.55,
            "latency_noise_scale": 0.35,
            "honest_connection_bias": 0.45 if strategy == "camouflage" else 0.25,
            "randomization": 0.15,
            "exploit_pressure": 0.30,
            "connection_expansion": 0.0,
            "burst_start_timestep": burst_start,
            "burst_active": False,
            "fraud_score": 0.0,
            "stealth_budget": 1.0,
            "max_stealth_budget": 1.0,
            "stealth_regen_rate": STEALTH_REGEN_RATE,
            "feature_priority": list(feature_priority),
            "last_targeted_feature": "",
            "target_action_counts": {
                "reduce_clustering": 0,
                "add_noise": 0,
                "increase_connections": 0,
            },
        }
    return states


def _non_linear_latency(distance_km: float, continent_penalty_ms: float, rng: np.random.Generator) -> float:
    baseline = 4.5 + 0.0105 * distance_km + 0.0002 * (distance_km**1.23)
    jitter = float(rng.normal(0.0, 4.0))
    routing_noise = float(rng.lognormal(mean=1.15, sigma=0.32) - np.exp(1.15))
    spike = float(rng.uniform(40.0, 130.0)) if rng.random() < 0.04 else 0.0
    return max(1.0, baseline + continent_penalty_ms + jitter + routing_noise + spike)


def _smart_state_for_edge(
    i: int,
    j: int,
    nodes: Sequence[NodeRecord],
    smart_states: Dict[int, Dict[str, object]],
) -> Dict[str, object] | None:
    ni = nodes[i]
    nj = nodes[j]
    if ni.label == "smart_attacker":
        return smart_states.get(ni.node_id)
    if nj.label == "smart_attacker":
        return smart_states.get(nj.node_id)
    return None


def _derive_feature_priority(
    model: object | None,
    feature_columns: Sequence[str],
    top_k: int = 3,
) -> List[str]:
    """Rank base features by model importance and return top-k actionable targets.

    Falls back to DEFAULT_TARGET_FEATURE_PRIORITY when model importances are unavailable.
    """
    if model is None or not feature_columns:
        return list(DEFAULT_TARGET_FEATURE_PRIORITY[:top_k])

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return list(DEFAULT_TARGET_FEATURE_PRIORITY[:top_k])

    base_scores: Dict[str, float] = {}
    for feature_name, importance in zip(feature_columns, importances):
        base = str(feature_name)
        for suffix in ("_mean", "_std", "_last"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        base_scores[base] = base_scores.get(base, 0.0) + float(importance)

    ranked = sorted(base_scores.items(), key=lambda item: item[1], reverse=True)
    selected = [name for name, _ in ranked if name in FEATURE_TARGET_ACTIONS][:top_k]
    if not selected:
        selected = list(DEFAULT_TARGET_FEATURE_PRIORITY[:top_k])
    return selected


def _activate_attack_scenarios(
    timestep: int,
    nodes: Sequence[NodeRecord],
    scenario_state: Dict[str, Any],
    rng: np.random.Generator,
) -> List[AttackScenarioEvent]:
    """Randomly activate regional attack, sybil swarm, and cascading trust failure scenarios."""
    events: List[AttackScenarioEvent] = []
    smart_ids = [n.node_id for n in nodes if n.label == "smart_attacker"]
    honest_ids = [n.node_id for n in nodes if n.label == "honest"]
    if not smart_ids:
        return events

    if rng.random() < REGIONAL_ATTACK_INJECTION_PROB:
        scenario_state["regional_attack_active"] = True
        scenario_state["regional_target_region"] = str(scenario_state["high_value_region"])
        target_region = str(scenario_state["regional_target_region"])
        impacted = sum(1 for n in nodes if n.label != "honest" and n.region == target_region)
        events.append(AttackScenarioEvent(timestep=timestep, scenario_name="regional_attack", impacted_node_count=impacted))
    else:
        scenario_state["regional_attack_active"] = False

    if rng.random() < SYBIL_SWARM_INJECTION_PROB:
        swarm_size = max(MIN_SYBIL_SWARM_SIZE, int(round(SYBIL_SWARM_FRACTION * len(smart_ids))))
        swarm = list(rng.choice(np.array(smart_ids, dtype=int), size=min(swarm_size, len(smart_ids)), replace=False))
        scenario_state["sybil_swarm_nodes"] = set(int(x) for x in swarm)
        events.append(
            AttackScenarioEvent(
                timestep=timestep,
                scenario_name="sybil_swarm",
                impacted_node_count=len(scenario_state["sybil_swarm_nodes"]),
            )
        )
    else:
        scenario_state["sybil_swarm_nodes"] = set()

    if rng.random() < CASCADING_FAILURE_INJECTION_PROB:
        existing = set(int(x) for x in scenario_state.get("corrupted_honest_nodes", set()))
        spread_count = max(MIN_CASCADING_SPREAD, int(round(CASCADING_SPREAD_FRACTION * len(honest_ids))))
        if honest_ids:
            new_targets = rng.choice(np.array(honest_ids, dtype=int), size=min(spread_count, len(honest_ids)), replace=False)
            for node_id in new_targets:
                existing.add(int(node_id))
        scenario_state["corrupted_honest_nodes"] = existing
        events.append(
            AttackScenarioEvent(
                timestep=timestep,
                scenario_name="cascading_trust_failure",
                impacted_node_count=len(existing),
            )
        )

    return events


def _apply_targeted_action(state: Dict[str, object], action: str, adaptation_strength: float, rng: np.random.Generator) -> None:
    """Apply one budgeted smart-attacker action to manipulate selected high-importance features."""
    if action == "reduce_clustering":
        state["clustering_bias"] = max(0.02, float(state["clustering_bias"]) - 0.18 * adaptation_strength)
    elif action == "add_noise":
        state["latency_noise_scale"] = min(3.0, float(state["latency_noise_scale"]) + 0.30 * adaptation_strength)
    elif action == "increase_connections":
        state["honest_connection_bias"] = min(1.0, float(state["honest_connection_bias"]) + 0.22 * adaptation_strength)
        state["randomization"] = min(1.0, float(state["randomization"]) + 0.18 * adaptation_strength)
        state["connection_expansion"] = min(1.0, float(state["connection_expansion"]) + float(rng.uniform(0.08, 0.18)))

def _build_latency_matrix(
    nodes: Sequence[NodeRecord],
    timestep: int,
    total_timesteps: int,
    rng: np.random.Generator,
    measurement_error_std: float,
    packet_loss_rate: float,
    missing_latency_rate: float,
    smart_states: Dict[int, Dict[str, object]],
    scenario_state: Dict[str, Any],
) -> np.ndarray:
    total_nodes = len(nodes)
    matrix = np.full((total_nodes, total_nodes), np.nan, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    corrupted_honest = set(int(x) for x in scenario_state.get("corrupted_honest_nodes", set()))
    sybil_nodes = set(int(x) for x in scenario_state.get("sybil_swarm_nodes", set()))
    regional_active = bool(scenario_state.get("regional_attack_active", False))
    regional_target = str(scenario_state.get("regional_target_region", ""))

    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            ni = nodes[i]
            nj = nodes[j]
            distance = haversine_km((ni.real_lat, ni.real_lon), (nj.real_lat, nj.real_lon))
            continent_penalty = _continent_penalty(ni.continent, nj.continent, rng)

            if ni.label == "naive_attacker" and nj.label == "naive_attacker":
                base = float(rng.uniform(0.2, 2.8))
            else:
                base = _non_linear_latency(distance, continent_penalty, rng)

            if i in corrupted_honest or j in corrupted_honest:
                base = max(MIN_CORRUPTED_LATENCY, base + float(rng.normal(CORRUPTED_LATENCY_MEAN, CORRUPTED_LATENCY_STD)))

            if i in sybil_nodes and j in sybil_nodes:
                base *= SYBIL_LATENCY_REDUCTION
                base = max(MIN_SYBIL_LATENCY, base + float(rng.normal(0.0, SYBIL_LATENCY_NOISE_STD)))

            if regional_active and (ni.region == regional_target or nj.region == regional_target):
                base *= REGIONAL_ATTACK_LATENCY_FACTOR

            smart_state = _smart_state_for_edge(i, j, nodes, smart_states)
            if smart_state is not None:
                adaptation = float(smart_state.get("adaptation_strength", 0.0))
                clustering_bias = float(smart_state.get("clustering_bias", 0.5))
                exploit_pressure = float(smart_state.get("exploit_pressure", 0.3))
                noise_scale = float(smart_state.get("latency_noise_scale", 0.3))
                strategy = str(smart_state.get("strategy", "low_and_slow"))
                burst_active = bool(smart_state.get("burst_active", False))

                if ni.label != "honest" and nj.label != "honest":
                    base *= max(0.20, 0.70 - 0.45 * clustering_bias - 0.20 * exploit_pressure)

                if strategy == "low_and_slow":
                    noise_scale *= 0.8
                    base *= 1.03
                elif strategy == "burst_attack" and burst_active:
                    base *= max(0.12, 0.45 - 0.25 * exploit_pressure)
                    noise_scale *= 1.35
                elif strategy == "camouflage":
                    base *= 1.06
                    noise_scale *= 1.1

                stealth_noise = float(rng.normal(0.0, 6.0 * (1.0 + noise_scale + adaptation)))
                base = max(1.0, base + stealth_noise)

                if rng.random() < min(0.65, 0.18 + 0.40 * adaptation):
                    base = _non_linear_latency(distance, continent_penalty * 0.9, rng)

            if ni.anomalous_honest or nj.anomalous_honest:
                base += float(rng.normal(0.0, 12.0))
                if rng.random() < 0.08:
                    base += float(rng.uniform(60.0, 180.0))

            latency_ij = max(1.0, base + float(rng.normal(0.0, 2.5)))
            latency_ji = max(1.0, base + float(rng.normal(0.0, 2.5)))

            if ni.unstable_connection and rng.random() < 0.10:
                latency_ij += float(rng.uniform(25.0, 110.0))
            if nj.unstable_connection and rng.random() < 0.10:
                latency_ji += float(rng.uniform(25.0, 110.0))

            latency_ij *= max(0.65, 1.0 + float(rng.normal(0.0, measurement_error_std)))
            latency_ji *= max(0.65, 1.0 + float(rng.normal(0.0, measurement_error_std)))

            if rng.random() < packet_loss_rate:
                latency_ij = np.nan
            if rng.random() < packet_loss_rate:
                latency_ji = np.nan
            if rng.random() < missing_latency_rate:
                latency_ij = np.nan
            if rng.random() < missing_latency_rate:
                latency_ji = np.nan

            matrix[i, j] = latency_ij
            matrix[j, i] = latency_ji

    return matrix


def _choose_peers(
    node_id: int,
    candidates: np.ndarray,
    latencies: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if candidates.size == 0:
        return np.array([], dtype=int)
    valid_mask = np.isfinite(latencies[candidates]) & (candidates != node_id)
    valid = candidates[valid_mask]
    if valid.size == 0:
        return np.array([], dtype=int)
    k_eff = min(k, valid.size)
    probs = 1.0 / np.maximum(latencies[valid], 1.0)
    probs = probs / probs.sum()
    return rng.choice(valid, size=k_eff, replace=False, p=probs)


def _build_peer_graph(
    nodes: Sequence[NodeRecord],
    latency_matrix: np.ndarray,
    timestep: int,
    total_timesteps: int,
    partial_visibility_rate: float,
    rng: np.random.Generator,
    smart_states: Dict[int, Dict[str, object]],
    scenario_state: Dict[str, Any],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    labels = np.array([n.label for n in nodes], dtype=object)
    all_ids = np.arange(len(nodes))
    naive_ids = np.where(labels == "naive_attacker")[0]
    attacker_ids = np.where(labels != "honest")[0]
    honest_ids = np.where(labels == "honest")[0]
    sybil_nodes = set(int(x) for x in scenario_state.get("sybil_swarm_nodes", set()))
    corrupted_honest = set(int(x) for x in scenario_state.get("corrupted_honest_nodes", set()))

    for node in nodes:
        graph.add_node(
            node.node_id,
            label=node.label,
            real_location=(node.real_lat, node.real_lon),
            claimed_location=(node.claimed_lat, node.claimed_lon),
            region=node.region,
            anomalous_honest=node.anomalous_honest,
            unstable_connection=node.unstable_connection,
            attacker_strategy=node.attacker_strategy,
        )

    smart_external_bias = 0.45 + 0.2 * (timestep / max(1, total_timesteps - 1))

    for node in nodes:
        i = node.node_id
        base_candidates = np.delete(all_ids, i)

        if node.label == "honest" and i not in corrupted_honest:
            k = int(rng.integers(7, 15))
            if node.unstable_connection:
                k = max(3, int(round(k * rng.uniform(0.45, 0.75))))
            peers = _choose_peers(i, base_candidates, latency_matrix[i], k, rng)
        elif node.label == "naive_attacker" or i in corrupted_honest:
            k = int(rng.integers(9, 16))
            mostly_attackers = int(round(k * rng.uniform(0.72, 0.9)))
            attacker_pool = np.setdiff1d(naive_ids, np.array([i]))
            if i in corrupted_honest:
                attacker_pool = np.setdiff1d(attacker_ids, np.array([i]))
            attacker_peers = _choose_peers(i, attacker_pool, latency_matrix[i], mostly_attackers, rng)
            remainder = k - attacker_peers.size
            other_pool = np.setdiff1d(base_candidates, naive_ids)
            other_peers = _choose_peers(i, other_pool, latency_matrix[i], remainder, rng)
            peers = np.unique(np.concatenate([attacker_peers, other_peers]))
        else:
            state = smart_states.get(i, {})
            strategy = str(state.get("strategy", "low_and_slow"))
            clustering_bias = float(state.get("clustering_bias", 0.55))
            honest_connection_bias = float(state.get("honest_connection_bias", 0.25))
            exploit_pressure = float(state.get("exploit_pressure", 0.30))
            randomization = float(state.get("randomization", 0.15))
            burst_active = bool(state.get("burst_active", False))

            if strategy == "low_and_slow":
                k = int(rng.integers(6, 12))
            else:
                k = int(rng.integers(8, 15))
            k = int(round(k * (1.0 + float(state.get("connection_expansion", 0.0)))))
            k = max(4, min(k, len(base_candidates)))

            attacker_ratio = float(
                np.clip(
                    0.55 + 0.35 * clustering_bias + 0.20 * exploit_pressure - 0.45 * honest_connection_bias,
                    0.05,
                    0.95,
                )
            )
            attacker_ratio *= max(0.6, 1.0 - 0.25 * randomization)

            if strategy == "low_and_slow":
                attacker_ratio *= 0.55
            elif strategy == "burst_attack":
                attacker_ratio *= 1.40 if burst_active else 0.45
            elif strategy == "camouflage":
                attacker_ratio *= 0.40
                honest_connection_bias = max(honest_connection_bias, 0.55)
            if i in sybil_nodes and sybil_nodes:
                attacker_ratio = min(MAX_SYBIL_ATTACKER_RATIO, attacker_ratio + SYBIL_ATTACKER_RATIO_BOOST)

            attacker_target = int(round(k * (1.0 - smart_external_bias) * attacker_ratio))
            external_target = max(0, k - attacker_target)
            attacker_pool = np.setdiff1d(attacker_ids, np.array([i]))
            if i in sybil_nodes and sybil_nodes:
                attacker_pool = np.array([nid for nid in sybil_nodes if nid != i], dtype=int)
            attacker_peers = _choose_peers(i, attacker_pool, latency_matrix[i], attacker_target, rng)
            external_pool = np.setdiff1d(honest_ids, np.array([i]))
            if scenario_state.get("regional_attack_active", False):
                target_region = str(scenario_state.get("regional_target_region", ""))
                regional_honest = [n.node_id for n in nodes if n.label == "honest" and n.region == target_region and n.node_id != i]
                if regional_honest:
                    external_pool = np.array(regional_honest, dtype=int)
            external_target = int(round(external_target * (1.0 + honest_connection_bias)))
            external_peers = _choose_peers(i, external_pool, latency_matrix[i], external_target, rng)
            peers = np.unique(np.concatenate([attacker_peers, external_peers]))

        if peers.size == 0:
            fallback = _choose_peers(i, base_candidates, latency_matrix[i], 3, rng)
            peers = fallback

        for p in peers:
            if rng.random() < partial_visibility_rate:
                continue
            latency = latency_matrix[i, int(p)]
            if not np.isfinite(latency):
                continue
            graph.add_edge(i, int(p), latency=float(latency))

    return graph


def _score_snapshot_for_adaptation(
    nodes: Sequence[NodeRecord],
    latency_matrix: np.ndarray,
    graph: nx.DiGraph,
    timestep: int,
    model: object,
    feature_columns: Sequence[str],
) -> Dict[int, float]:
    from features import build_model_features_from_temporal_frame, extract_snapshot_features

    nodes_meta = {
        n.node_id: {
            "label": n.label,
            "claimed_location": (n.claimed_lat, n.claimed_lon),
        }
        for n in nodes
    }
    snapshot_df = extract_snapshot_features(
        graph=graph,
        latency_matrix=latency_matrix,
        nodes_meta=nodes_meta,
        timestep=timestep,
    )
    model_df = build_model_features_from_temporal_frame(snapshot_df)
    X = model_df[list(feature_columns)]
    scores = model.predict_proba(X)[:, 1]
    return {int(node_id): float(score) for node_id, score in zip(model_df["node_id"], scores)}


def _update_smart_attacker_states(
    smart_states: Dict[int, Dict[str, object]],
    fraud_scores: Dict[int, float],
    threshold: float,
    timestep: int,
    total_timesteps: int,
    adaptation_base: float,
    adaptation_growth: float,
    rng: np.random.Generator,
) -> None:
    for node_id, state in smart_states.items():
        strategy = str(state.get("strategy", "low_and_slow"))
        adaptation_strength = adaptation_base + adaptation_growth * (
            timestep / max(1, total_timesteps - 1)
        )
        score = float(fraud_scores.get(node_id, 0.0))
        state["fraud_score"] = score
        state["adaptation_strength"] = adaptation_strength

        state["burst_active"] = strategy == "burst_attack" and timestep >= int(state.get("burst_start_timestep", 0))
        state["connection_expansion"] = max(0.0, float(state.get("connection_expansion", 0.0)) - 0.03)

        max_budget = float(state.get("max_stealth_budget", 1.0))
        regen = float(state.get("stealth_regen_rate", STEALTH_REGEN_RATE))
        budget = min(max_budget, float(state.get("stealth_budget", max_budget)) + regen)
        state["stealth_budget"] = budget
        state["last_targeted_feature"] = ""

        if score >= threshold and budget > 0.0:
            priorities = list(state.get("feature_priority", list(DEFAULT_TARGET_FEATURE_PRIORITY)))
            for feature_name in priorities:
                action = FEATURE_TARGET_ACTIONS.get(str(feature_name))
                if action is None:
                    continue
                cost = float(STEALTH_ACTION_COSTS[action])
                if budget < cost:
                    continue
                _apply_targeted_action(state=state, action=action, adaptation_strength=adaptation_strength, rng=rng)
                budget -= cost
                state["stealth_budget"] = budget
                state["last_targeted_feature"] = str(feature_name)
                action_counts = dict(state.get("target_action_counts", {}))
                action_counts[action] = int(action_counts.get(action, 0)) + 1
                state["target_action_counts"] = action_counts
                state["exploit_pressure"] = max(0.0, float(state["exploit_pressure"]) - 0.10 * adaptation_strength)
                break
        else:
            state["clustering_bias"] = min(1.0, float(state["clustering_bias"]) + 0.08 * adaptation_strength)
            state["latency_noise_scale"] = max(0.0, float(state["latency_noise_scale"]) - 0.05 * adaptation_strength)
            state["honest_connection_bias"] = max(0.0, float(state["honest_connection_bias"]) - 0.07 * adaptation_strength)
            state["randomization"] = max(0.0, float(state["randomization"]) - 0.05 * adaptation_strength)
            state["exploit_pressure"] = min(1.0, float(state["exploit_pressure"]) + 0.12 * adaptation_strength)

        if strategy == "low_and_slow":
            state["exploit_pressure"] = min(float(state["exploit_pressure"]), 0.45)
            state["clustering_bias"] = min(float(state["clustering_bias"]), 0.50)
        elif strategy == "camouflage":
            state["honest_connection_bias"] = max(float(state["honest_connection_bias"]), 0.55)
        elif strategy == "burst_attack" and bool(state["burst_active"]) and score < threshold:
            state["exploit_pressure"] = min(1.0, float(state["exploit_pressure"]) + 0.25 + float(rng.uniform(0.0, 0.2)))


def build_network_simulation(
    total_nodes: int = 200,
    honest_ratio: float = 0.85,
    naive_attacker_ratio: float = 0.105,
    smart_attacker_ratio: float = 0.045,
    attacker_cluster_center: Coordinate = (1.3521, 103.8198),
    attacker_cluster_continent: str = "asia",
    honest_anomaly_rate: float = 0.12,
    honest_unstable_rate: float = 0.10,
    measurement_error_std: float = 0.05,
    missing_latency_rate: float = 0.03,
    packet_loss_rate: float = 0.04,
    partial_visibility_rate: float = 0.06,
    time_steps: int = 15,
    seed: int = 42,
    adaptation_base: float = SMART_ATTACKER_BASE_ADAPTATION,
    adaptation_growth: float = SMART_ATTACKER_ADAPTATION_GROWTH,
    model_for_adaptation: object | None = None,
    fraud_score_threshold: float = 0.5,
    feature_columns: Sequence[str] | None = None,
    smart_strategy_mix: Dict[str, float] | None = None,
    smart_target_top_k: int = 3,
    enable_system_attacks: bool = True,
) -> SimulationResult:
    if total_nodes < 10:
        raise ValueError("total_nodes must be at least 10 for a meaningful topology")
    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")

    ratio_sum = honest_ratio + naive_attacker_ratio + smart_attacker_ratio
    if not np.isclose(ratio_sum, 1.0, atol=RATIO_SUM_TOLERANCE):
        raise ValueError("Node-type ratios must sum to 1.0")

    rng = np.random.default_rng(seed)

    honest_count = int(round(total_nodes * honest_ratio))
    naive_count = int(round(total_nodes * naive_attacker_ratio))
    smart_count = total_nodes - honest_count - naive_count

    if smart_count < 1:
        deficit = 1 - smart_count
        honest_reducible = max(0, honest_count - 1)
        reduce_from_honest = min(honest_reducible, deficit)
        honest_count -= reduce_from_honest
        deficit -= reduce_from_honest
        if deficit > 0:
            naive_count = max(0, naive_count - deficit)
        smart_count = 1

    if honest_count + naive_count + smart_count != total_nodes:
        smart_count = total_nodes - honest_count - naive_count
    if smart_count < 1:
        raise ValueError("Node counts are infeasible for the requested ratios and total_nodes")

    ids = np.arange(total_nodes)
    honest_ids = ids[:honest_count]
    naive_ids = ids[honest_count : honest_count + naive_count]
    smart_ids = ids[honest_count + naive_count :]
    smart_strategies = _sample_smart_attacker_strategies(smart_ids, rng, strategy_mix=smart_strategy_mix)
    high_value_regions = ["new_york", "tokyo", "london", "singapore"]
    high_value_region = str(high_value_regions[int(rng.integers(0, len(high_value_regions)))])

    nodes = (
        _sample_honest_nodes(honest_ids, rng, anomaly_rate=honest_anomaly_rate, unstable_rate=honest_unstable_rate)
        + _sample_naive_attackers(
            naive_ids,
            attacker_cluster_center,
            attacker_cluster_continent,
            rng,
            target_region=high_value_region,
        )
        + _sample_smart_attackers(smart_ids, smart_strategies, rng, target_region=high_value_region)
    )
    feature_columns = tuple(feature_columns or [])
    feature_priority = _derive_feature_priority(model_for_adaptation, feature_columns, top_k=smart_target_top_k)
    smart_states = _initialize_smart_states(nodes, total_timesteps=time_steps, rng=rng, feature_priority=feature_priority)
    scenario_state: Dict[str, Any] = {
        "high_value_region": high_value_region,
        "regional_attack_active": False,
        "regional_target_region": high_value_region,
        "sybil_swarm_nodes": set(),
        "corrupted_honest_nodes": set(),
    }

    snapshots: List[TimeStepSnapshot] = []
    fraud_scores_over_time: List[Dict[int, float]] = []
    scenario_events: List[AttackScenarioEvent] = []

    for timestep in range(time_steps):
        if enable_system_attacks:
            scenario_events.extend(_activate_attack_scenarios(timestep=timestep, nodes=nodes, scenario_state=scenario_state, rng=rng))
        latency_matrix = _build_latency_matrix(
            nodes=nodes,
            timestep=timestep,
            total_timesteps=time_steps,
            rng=rng,
            measurement_error_std=measurement_error_std,
            packet_loss_rate=packet_loss_rate,
            missing_latency_rate=missing_latency_rate,
            smart_states=smart_states,
            scenario_state=scenario_state,
        )
        peer_graph = _build_peer_graph(
            nodes=nodes,
            latency_matrix=latency_matrix,
            timestep=timestep,
            total_timesteps=time_steps,
            partial_visibility_rate=partial_visibility_rate,
            rng=rng,
            smart_states=smart_states,
            scenario_state=scenario_state,
        )

        fraud_scores: Dict[int, float] = {}
        if model_for_adaptation is not None and feature_columns:
            fraud_scores = _score_snapshot_for_adaptation(
                nodes=nodes,
                latency_matrix=latency_matrix,
                graph=peer_graph,
                timestep=timestep,
                model=model_for_adaptation,
                feature_columns=feature_columns,
            )
        fraud_scores_over_time.append(fraud_scores)

        snapshots.append(
            TimeStepSnapshot(
                timestep=timestep,
                latency_matrix=latency_matrix,
                peer_graph=peer_graph,
                fraud_scores=fraud_scores,
            )
        )

        if smart_states and timestep < time_steps - 1:
            _update_smart_attacker_states(
                smart_states=smart_states,
                fraud_scores=fraud_scores,
                threshold=fraud_score_threshold,
                timestep=timestep,
                total_timesteps=time_steps,
                adaptation_base=adaptation_base,
                adaptation_growth=adaptation_growth,
                rng=rng,
            )

    final_snapshot = snapshots[-1]
    event_counts: Dict[str, float] = {
        "regional_attack_events": 0.0,
        "sybil_swarm_events": 0.0,
        "cascading_trust_failure_events": 0.0,
    }
    for event in scenario_events:
        key = f"{event.scenario_name}_events"
        event_counts[key] = float(event_counts.get(key, 0.0) + 1.0)
    if smart_states:
        avg_remaining_budget = float(np.mean([float(state.get("stealth_budget", 0.0)) for state in smart_states.values()]))
    else:
        avg_remaining_budget = 0.0
    scenario_metrics: Dict[str, float] = {
        **event_counts,
        "scenario_event_count": float(len(scenario_events)),
        "high_value_region_node_count": float(sum(1 for n in nodes if n.region == high_value_region)),
        "corrupted_honest_node_count": float(len(scenario_state.get("corrupted_honest_nodes", set()))),
        "avg_smart_attacker_remaining_stealth_budget": avg_remaining_budget,
    }
    return SimulationResult(
        nodes=nodes,
        latency_matrix=final_snapshot.latency_matrix,
        peer_graph=final_snapshot.peer_graph,
        time_steps=snapshots,
        fraud_scores_over_time=fraud_scores_over_time,
        scenario_events=scenario_events,
        scenario_metrics=scenario_metrics,
    )
