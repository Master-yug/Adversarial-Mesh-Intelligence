from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from simulation.geo import haversine_km
from defender_engine.policies import (
    AdaptiveDefenderPolicy,
    StrategicAttackerPolicy,
    predict_fraud_and_uncertainty,
    summarize_defender_metrics,
)
from attacker_engine.strategies import (
    StrategyLearningState,
    evolve_strategy_mix,
    normalize_strategy_mix,
    update_learning_state,
)

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
SMART_ATTACKER_STRATEGIES = (
    "low_and_slow",
    "burst_attack",
    "camouflage",
    "perfect_mimic",
    "slow_drift",
    "decoy_attacker",
    "mixed_cluster",
)
SMART_ATTACKER_STRATEGY_ARRAY = np.array(SMART_ATTACKER_STRATEGIES, dtype=object)
DEFAULT_SMART_STRATEGY_MIX = {
    "low_and_slow": 0.22,
    "burst_attack": 0.16,
    "camouflage": 0.18,
    "perfect_mimic": 0.14,
    "slow_drift": 0.14,
    "decoy_attacker": 0.08,
    "mixed_cluster": 0.08,
}
WARMUP_REQUIRED_STRATEGIES = {"perfect_mimic", "slow_drift", "mixed_cluster"}
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
IDENTITY_RESET_HIGH_SCORE_FACTOR = 1.15
IDENTITY_RESET_LOW_REWARD_THRESHOLD = 0.55
IDENTITY_RESET_COOLDOWN_TIMESTEPS = 2
NO_RESET_SENTINEL = -999
HIGH_PROFIT_EXPLOIT_THRESHOLD = 2.2
DEFENSIVE_UNCERTAINTY_THRESHOLD = 0.45
DETECTED_ATTACKER_REWARD = 0.55
UNDETECTED_ATTACKER_REWARD = 2.4
RESIDUAL_DETECTED_ATTACKER_LEAK = 0.22
DELAYED_SIGNAL_MIX = 0.65
MAX_OBSERVATION_LAG = 2
DETECTION_PENALTY = 1.15
MIN_CAMPAIGN_SIZE = 3
MAX_CAMPAIGN_SIZE = 8
ROLE_POOL = ("leader", "decoy", "executor")
EXECUTOR_SYNC_WEIGHT = 0.22
DEFAULT_CAMPAIGN_SYNC_WEIGHT = 0.15
STRATEGY_RESAMPLING_RATE = 0.38

Difficulties = ("easy", "medium", "hard", "extreme")
DIFFICULTY_PROFILES: Dict[str, Dict[str, float]] = {
    "easy": {
        "noise_multiplier": 0.75,
        "attacker_intelligence": 0.55,
        "visibility_multiplier": 1.10,
        "label_corruption_multiplier": 0.70,
        "partial_label_rate": 0.10,
        "delayed_label_steps": 1.0,
        "suboptimal_action_rate": 0.30,
        "decision_epsilon": 0.18,
        "feedback_noise_std": 0.11,
        "feedback_delay_steps": 2.0,
        "strategy_mutation_rate": 0.03,
    },
    "medium": {
        "noise_multiplier": 1.00,
        "attacker_intelligence": 0.72,
        "visibility_multiplier": 1.00,
        "label_corruption_multiplier": 1.00,
        "partial_label_rate": 0.20,
        "delayed_label_steps": 2.0,
        "suboptimal_action_rate": 0.22,
        "decision_epsilon": 0.14,
        "feedback_noise_std": 0.08,
        "feedback_delay_steps": 2.0,
        "strategy_mutation_rate": 0.06,
    },
    "hard": {
        "noise_multiplier": 1.20,
        "attacker_intelligence": 0.86,
        "visibility_multiplier": 0.88,
        "label_corruption_multiplier": 1.35,
        "partial_label_rate": 0.32,
        "delayed_label_steps": 3.0,
        "suboptimal_action_rate": 0.16,
        "decision_epsilon": 0.10,
        "feedback_noise_std": 0.05,
        "feedback_delay_steps": 3.0,
        "strategy_mutation_rate": 0.11,
    },
    "extreme": {
        "noise_multiplier": 1.38,
        "attacker_intelligence": 0.94,
        "visibility_multiplier": 0.76,
        "label_corruption_multiplier": 1.65,
        "partial_label_rate": 0.45,
        "delayed_label_steps": 4.0,
        "suboptimal_action_rate": 0.10,
        "decision_epsilon": 0.08,
        "feedback_noise_std": 0.03,
        "feedback_delay_steps": 4.0,
        "strategy_mutation_rate": 0.16,
    },
}
DEFENDER_BUDGET_FRACTION_BY_DIFFICULTY: Dict[str, float] = {
    "easy": 0.09,
    "medium": 0.07,
    "hard": 0.05,
    "extreme": 0.04,
}


def _difficulty_profile(level: str) -> Dict[str, float]:
    key = str(level).strip().lower()
    if key not in DIFFICULTY_PROFILES:
        raise ValueError(f"difficulty_level must be one of {Difficulties}")
    return dict(DIFFICULTY_PROFILES[key])


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
    gray_zone: bool = False
    ambiguity_anchor: bool = False
    visibility_variance: float = 0.0


@dataclass(frozen=True)
class TimeStepSnapshot:
    timestep: int
    latency_matrix: np.ndarray
    peer_graph: nx.DiGraph
    fraud_scores: Dict[int, float]
    uncertainty_scores: Dict[int, float] = field(default_factory=dict)


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
    uncertainty_over_time: List[Dict[int, float]]
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
    gray_zone_rate: float,
    ambiguity_floor_rate: float,
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
        is_gray_zone = bool(rng.random() < gray_zone_rate)
        ambiguity_anchor = bool(rng.random() < ambiguity_floor_rate)
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
                gray_zone=is_gray_zone,
                ambiguity_anchor=ambiguity_anchor or is_gray_zone,
                visibility_variance=float(rng.uniform(0.0, 0.35 if is_gray_zone else 0.20)),
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
    mix = dict(strategy_mix or DEFAULT_SMART_STRATEGY_MIX)
    if "decoy" in mix:
        mix["decoy_attacker"] = float(mix.get("decoy_attacker", 0.0)) + float(mix["decoy"])
        mix.pop("decoy", None)
    probs = np.array([float(mix.get(strategy, 0.0)) for strategy in SMART_ATTACKER_STRATEGIES], dtype=float)
    if probs.sum() <= 0:
        probs = np.array([1.0 / len(SMART_ATTACKER_STRATEGIES)] * len(SMART_ATTACKER_STRATEGIES), dtype=float)
    probs = probs / probs.sum()
    strategies = rng.choice(SMART_ATTACKER_STRATEGIES, size=len(node_ids), replace=True, p=probs)
    return {int(node_id): str(strategy) for node_id, strategy in zip(node_ids, strategies)}


def _compose_smart_strategy_mix(
    strategy_mix: Dict[str, float] | None,
    attacker_sophistication: float,
) -> Dict[str, float]:
    mix = normalize_strategy_mix(strategy_mix or DEFAULT_SMART_STRATEGY_MIX)
    stealth = float(np.clip(attacker_sophistication, 0.0, 1.0))
    mix["perfect_mimic"] = float(mix.get("perfect_mimic", 0.0) + 0.22 * stealth)
    mix["slow_drift"] = float(mix.get("slow_drift", 0.0) + 0.18 * stealth)
    mix["mixed_cluster"] = float(mix.get("mixed_cluster", 0.0) + 0.14 * stealth)
    mix["decoy_attacker"] = float(mix.get("decoy_attacker", 0.0) + float(mix.get("decoy", 0.0)) + 0.15 * stealth)
    mix.pop("decoy", None)
    mix["burst_attack"] = float(max(0.01, mix.get("burst_attack", 0.0) - 0.20 * stealth))
    mix["low_and_slow"] = float(max(0.02, mix.get("low_and_slow", 0.0) - 0.08 * stealth))
    total = float(sum(max(v, 0.0) for v in mix.values()))
    if total <= 0.0:
        return normalize_strategy_mix(DEFAULT_SMART_STRATEGY_MIX)
    return normalize_strategy_mix({k: float(max(v, 0.0) / total) for k, v in mix.items()})


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
        fake_probability = 0.12 if strategy in {"low_and_slow", "perfect_mimic"} else 0.30
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
                gray_zone=False,
                ambiguity_anchor=bool(strategy in {"perfect_mimic", "mixed_cluster"}),
                visibility_variance=float(rng.uniform(0.05, 0.30)),
            )
        )
    return rows


def _initialize_smart_states(
    nodes: Sequence[NodeRecord],
    total_timesteps: int,
    rng: np.random.Generator,
    feature_priority: Sequence[str],
    attacker_sophistication: float,
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
        warmup_steps = int(
            rng.integers(
                2,
                max(3, min(total_timesteps, 7)),
            )
        )
        states[node.node_id] = {
            "strategy": strategy,
            "adaptation_strength": SMART_ATTACKER_BASE_ADAPTATION,
            "clustering_bias": 0.55,
            "latency_noise_scale": float(max(0.05, 0.35 - 0.15 * attacker_sophistication)),
            "honest_connection_bias": float(
                np.clip(
                    (0.45 if strategy == "camouflage" else 0.25) + 0.25 * attacker_sophistication,
                    0.0,
                    1.0,
                )
            ),
            "randomization": float(np.clip(0.15 + 0.18 * attacker_sophistication, 0.0, 1.0)),
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
                "noop": 0,
            },
            "cumulative_reward": 0.0,
            "reward_history": [],
            "public_node_id": f"node-{node.node_id}",
            "identity_reset_count": 0,
            "last_identity_reset_timestep": NO_RESET_SENTINEL,
            "post_reset_detections": 0,
            "detected_since_reset": False,
            "warmup_steps": warmup_steps if strategy in WARMUP_REQUIRED_STRATEGIES else 1,
            "drift_strength": 0.0 if strategy == "slow_drift" else 1.0,
            "is_decoy": bool(strategy == "decoy_attacker" and rng.random() < 0.55),
            "blend_probability": float(
                0.18
                + (
                    0.22
                    if strategy in {"perfect_mimic", "mixed_cluster", "decoy_attacker", "slow_drift"}
                    else 0.0
                )
                + float(rng.uniform(0.0, 0.10))
            ),
            "signal_history": [],
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


def _initialize_attack_campaigns(
    smart_states: Dict[int, Dict[str, object]],
    rng: np.random.Generator,
) -> Dict[int, Dict[str, object]]:
    smart_ids = list(smart_states.keys())
    if not smart_ids:
        return {}
    campaigns: Dict[int, Dict[str, object]] = {}
    rng.shuffle(smart_ids)
    campaign_id = 0
    index = 0
    while index < len(smart_ids):
        size = int(rng.integers(MIN_CAMPAIGN_SIZE, MAX_CAMPAIGN_SIZE + 1))
        group = smart_ids[index : index + size]
        if len(group) < MIN_CAMPAIGN_SIZE:
            group = smart_ids[max(0, len(smart_ids) - MIN_CAMPAIGN_SIZE) :]
        index += max(len(group), 1)
        if not group:
            break
        leader = int(group[0])
        roles: Dict[int, str] = {}
        for pos, node_id in enumerate(group):
            if pos == 0:
                role = "leader"
            else:
                role = str(rng.choice(np.array(ROLE_POOL[1:], dtype=object)))
            roles[int(node_id)] = role
            smart_states[int(node_id)]["campaign_id"] = campaign_id
            smart_states[int(node_id)]["campaign_role"] = role
            smart_states[int(node_id)]["campaign_leader"] = leader
        campaigns[campaign_id] = {"members": list(group), "leader": leader, "roles": roles}
        campaign_id += 1
    return campaigns


def _strategy_rewards_from_states(
    smart_states: Dict[int, Dict[str, object]],
    threshold: float,
) -> Dict[str, float]:
    grouped_rewards: Dict[str, List[float]] = {}
    for state in smart_states.values():
        strategy = str(state.get("strategy", "low_and_slow"))
        score = float(np.clip(state.get("fraud_score", 0.0), 0.0, 1.0))
        detected = score >= threshold
        latest_profit = float(state.get("latest_reward", 0.0))
        reward = latest_profit - (DETECTION_PENALTY if detected else 0.0)
        grouped_rewards.setdefault(strategy, []).append(float(reward))
    return {
        strategy: float(np.mean(values))
        for strategy, values in grouped_rewards.items()
        if values
    }


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
    delayed_observation_rate: float,
    noise_spike_rate: float,
    smart_states: Dict[int, Dict[str, object]],
    scenario_state: Dict[str, Any],
    lag_reference_matrices: Sequence[np.ndarray] | None = None,
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
                elif strategy == "perfect_mimic":
                    base = _non_linear_latency(distance, continent_penalty, rng) * float(rng.uniform(0.96, 1.06))
                    noise_scale *= 0.55
                elif strategy == "slow_drift":
                    warmup = int(smart_state.get("warmup_steps", 2))
                    drift_phase = max(0.0, (timestep - warmup) / max(total_timesteps - warmup, 1))
                    base *= float(1.04 - 0.33 * drift_phase)
                    noise_scale *= float(0.75 + 0.70 * drift_phase)
                elif strategy == "decoy_attacker":
                    if bool(smart_state.get("is_decoy", False)):
                        base *= float(rng.uniform(0.20, 0.55))
                        noise_scale *= 1.8
                    else:
                        base *= float(rng.uniform(0.92, 1.12))
                        noise_scale *= 0.8
                elif strategy == "mixed_cluster":
                    base *= float(rng.uniform(0.85, 1.05))
                    noise_scale *= 0.9

                stealth_noise = float(rng.normal(0.0, 6.0 * (1.0 + noise_scale + adaptation)))
                base = max(1.0, base + stealth_noise)

                if rng.random() < min(0.65, 0.18 + 0.40 * adaptation):
                    base = _non_linear_latency(distance, continent_penalty * 0.9, rng)

            if ni.anomalous_honest or nj.anomalous_honest:
                base += float(rng.normal(0.0, 12.0))
                if rng.random() < 0.08:
                    base += float(rng.uniform(60.0, 180.0))
            if ni.gray_zone or nj.gray_zone:
                base += float(rng.normal(0.0, 18.0))
                if rng.random() < 0.22:
                    base *= float(rng.uniform(0.45, 1.45))
            if ni.ambiguity_anchor or nj.ambiguity_anchor:
                base += float(rng.normal(0.0, 8.0))

            latency_ij = max(1.0, base + float(rng.normal(0.0, 2.5)))
            latency_ji = max(1.0, base + float(rng.normal(0.0, 2.5)))

            if ni.unstable_connection and rng.random() < 0.10:
                latency_ij += float(rng.uniform(25.0, 110.0))
            if nj.unstable_connection and rng.random() < 0.10:
                latency_ji += float(rng.uniform(25.0, 110.0))

            latency_ij *= max(0.65, 1.0 + float(rng.normal(0.0, measurement_error_std)))
            latency_ji *= max(0.65, 1.0 + float(rng.normal(0.0, measurement_error_std)))
            if rng.random() < noise_spike_rate:
                latency_ij += float(rng.normal(0.0, 45.0))
            if rng.random() < noise_spike_rate:
                latency_ji += float(rng.normal(0.0, 45.0))

            if lag_reference_matrices and rng.random() < delayed_observation_rate:
                lag = lag_reference_matrices[int(rng.integers(0, len(lag_reference_matrices)))]
                lag_ij = float(lag[i, j])
                if np.isfinite(lag_ij):
                    latency_ij = lag_ij
            if lag_reference_matrices and rng.random() < delayed_observation_rate:
                lag = lag_reference_matrices[int(rng.integers(0, len(lag_reference_matrices)))]
                lag_ji = float(lag[j, i])
                if np.isfinite(lag_ji):
                    latency_ji = lag_ji

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
    quarantined_nodes: set[int] | None = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    labels = np.array([n.label for n in nodes], dtype=object)
    all_ids = np.arange(len(nodes))
    naive_ids = np.where(labels == "naive_attacker")[0]
    attacker_ids = np.where(labels != "honest")[0]
    honest_ids = np.where(labels == "honest")[0]
    sybil_nodes = set(int(x) for x in scenario_state.get("sybil_swarm_nodes", set()))
    corrupted_honest = set(int(x) for x in scenario_state.get("corrupted_honest_nodes", set()))
    quarantined_nodes = quarantined_nodes or set()

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
            gray_zone=node.gray_zone,
            ambiguity_anchor=node.ambiguity_anchor,
        )

    smart_external_bias = 0.45 + 0.2 * (timestep / max(1, total_timesteps - 1))

    for node in nodes:
        i = node.node_id
        if i in quarantined_nodes:
            continue
        base_candidates = np.delete(all_ids, i)
        if quarantined_nodes:
            base_candidates = np.array([c for c in base_candidates if int(c) not in quarantined_nodes], dtype=int)

        if node.label == "honest" and i not in corrupted_honest:
            k = int(rng.integers(7, 15))
            if node.unstable_connection:
                k = max(3, int(round(k * rng.uniform(0.45, 0.75))))
            if node.gray_zone:
                k = max(4, int(round(k * rng.uniform(0.65, 1.15))))
            if (node.anomalous_honest or node.gray_zone or node.ambiguity_anchor) and rng.random() < 0.24:
                attacker_like_k = max(3, int(round(k * rng.uniform(0.55, 1.20))))
                attacker_pool = np.setdiff1d(attacker_ids, np.array([i]))
                attacker_target = int(round(attacker_like_k * rng.uniform(0.30, 0.62)))
                attacker_peers = _choose_peers(i, attacker_pool, latency_matrix[i], attacker_target, rng)
                honest_remainder = attacker_like_k - attacker_peers.size
                honest_pool = np.setdiff1d(honest_ids, np.array([i]))
                honest_peers = _choose_peers(i, honest_pool, latency_matrix[i], honest_remainder, rng)
                peers = np.unique(np.concatenate([attacker_peers, honest_peers]))
            else:
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
            warmup_steps = int(state.get("warmup_steps", 1))
            clustering_bias = float(state.get("clustering_bias", 0.55))
            honest_connection_bias = float(state.get("honest_connection_bias", 0.25))
            exploit_pressure = float(state.get("exploit_pressure", 0.30))
            randomization = float(state.get("randomization", 0.15))
            burst_active = bool(state.get("burst_active", False))
            in_warmup = timestep < warmup_steps

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
            elif strategy == "perfect_mimic":
                attacker_ratio *= 0.25
                honest_connection_bias = max(honest_connection_bias, 0.70)
            elif strategy == "slow_drift":
                phase = max(0.0, (timestep - warmup_steps) / max(total_timesteps - warmup_steps, 1))
                attacker_ratio *= float(0.35 + 1.10 * phase)
            elif strategy == "decoy_attacker":
                attacker_ratio *= 1.45 if bool(state.get("is_decoy", False)) else 0.30
            elif strategy == "mixed_cluster":
                attacker_ratio *= 0.55
                honest_connection_bias = max(honest_connection_bias, 0.55)
            if rng.random() < float(state.get("blend_probability", 0.2)):
                attacker_ratio *= float(rng.uniform(0.35, 0.78))
                honest_connection_bias = min(1.0, max(honest_connection_bias, float(rng.uniform(0.55, 0.92))))
                randomization = min(1.0, randomization + float(rng.uniform(0.08, 0.22)))

            if in_warmup:
                attacker_ratio *= 0.35
                honest_connection_bias = min(1.0, honest_connection_bias + 0.25)
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
            visibility_drop = partial_visibility_rate + node.visibility_variance * float(rng.uniform(0.0, 0.9))
            if node.gray_zone:
                visibility_drop += 0.06
            if rng.random() < visibility_drop:
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
) -> tuple[Dict[int, float], Dict[int, float], pd.DataFrame]:
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
    scores, uncertainty = predict_fraud_and_uncertainty(model, X)
    score_map = {int(node_id): float(score) for node_id, score in zip(model_df["node_id"], scores)}
    uncertainty_map = {int(node_id): float(u) for node_id, u in zip(model_df["node_id"], uncertainty)}
    return score_map, uncertainty_map, model_df


def _update_smart_attacker_states(
    smart_states: Dict[int, Dict[str, object]],
    campaigns: Dict[int, Dict[str, object]],
    fraud_scores: Dict[int, float],
    uncertainty_scores: Dict[int, float],
    model_df: pd.DataFrame,
    model_for_adaptation: object,
    feature_columns: Sequence[str],
    attacker_policy: StrategicAttackerPolicy,
    threshold: float,
    timestep: int,
    total_timesteps: int,
    adaptation_base: float,
    adaptation_growth: float,
    suboptimal_action_rate: float,
    decision_epsilon: float,
    feedback_noise_std: float,
    feedback_delay_steps: int,
    rng: np.random.Generator,
) -> None:
    feature_rows = {int(row["node_id"]): row for _, row in model_df.iterrows()}
    for node_id, state in smart_states.items():
        strategy = str(state.get("strategy", "low_and_slow"))
        adaptation_strength = adaptation_base + adaptation_growth * (
            timestep / max(1, total_timesteps - 1)
        )
        score = float(fraud_scores.get(node_id, 0.0))
        uncertainty = float(uncertainty_scores.get(node_id, 0.0))
        signal_history = [float(v) for v in state.get("signal_history", [])]
        signal_history.append(score)
        signal_history = signal_history[-12:]
        state["signal_history"] = signal_history
        delayed_idx = max(0, len(signal_history) - 1 - max(int(feedback_delay_steps), 0))
        delayed_score = float(signal_history[delayed_idx]) if signal_history else score
        perceived_score = float(np.clip(delayed_score + rng.normal(0.0, feedback_noise_std), 0.0, 1.0))
        state["fraud_score"] = score
        state["perceived_fraud_score"] = perceived_score
        state["adaptation_strength"] = adaptation_strength
        campaign_id = state.get("campaign_id")
        if campaign_id is not None and campaign_id in campaigns:
            campaign_role = str(state.get("campaign_role", "executor"))
            campaign = campaigns[campaign_id]
            leader_id = int(campaign.get("leader", node_id))
            leader_score = float(fraud_scores.get(leader_id, score))
            leader_strategy = str(smart_states.get(leader_id, {}).get("strategy", strategy))
            role_synchronization = EXECUTOR_SYNC_WEIGHT if campaign_role == "executor" else DEFAULT_CAMPAIGN_SYNC_WEIGHT
            perceived_score = float(np.clip((1.0 - role_synchronization) * perceived_score + role_synchronization * leader_score, 0.0, 1.0))
            state["perceived_fraud_score"] = perceived_score
            if campaign_role != "leader" and rng.random() < 0.30:
                state["strategy"] = leader_strategy
        else:
            campaign_role = "executor"

        state["burst_active"] = strategy == "burst_attack" and timestep >= int(state.get("burst_start_timestep", 0))
        state["connection_expansion"] = max(0.0, float(state.get("connection_expansion", 0.0)) - 0.03)

        max_budget = float(state.get("max_stealth_budget", 1.0))
        regen = float(state.get("stealth_regen_rate", STEALTH_REGEN_RATE))
        budget = min(max_budget, float(state.get("stealth_budget", max_budget)) + regen)
        state["stealth_budget"] = budget
        state["last_targeted_feature"] = ""
        rewards = [float(v) for v in state.get("reward_history", [])]
        recent_profit = float(np.mean(rewards[-3:])) if rewards else 0.0
        node_row = feature_rows.get(node_id)
        selected_action = "noop"
        if node_row is not None and model_for_adaptation is not None and feature_columns:
            decision = attacker_policy.decide_action(
                feature_row=node_row,
                feature_columns=feature_columns,
                model_bundle=model_for_adaptation,
                current_score=perceived_score,
                threshold=threshold,
                recent_profit=recent_profit,
            )
            selected_action = decision.action
            if campaign_role == "decoy" and selected_action != "add_noise":
                if "add_noise" in STEALTH_ACTION_COSTS:
                    selected_action = "add_noise"
            if campaign_role == "leader" and selected_action == "noop" and perceived_score >= threshold:
                selected_action = "reduce_clustering"
            if rng.random() < float(np.clip(decision_epsilon, 0.0, 1.0)):
                selected_action = str(rng.choice(np.array(list(STEALTH_ACTION_COSTS.keys()), dtype=object)))
            if rng.random() < float(np.clip(suboptimal_action_rate, 0.0, 1.0)):
                non_best = [a for a in STEALTH_ACTION_COSTS if a != decision.action]
                if non_best:
                    selected_action = str(rng.choice(np.array(non_best, dtype=object)))
            action_cost = float(STEALTH_ACTION_COSTS.get(selected_action, 0.0))
            if action_cost <= budget and selected_action != "noop":
                _apply_targeted_action(state=state, action=selected_action, adaptation_strength=adaptation_strength, rng=rng)
                budget -= action_cost
                state["stealth_budget"] = budget
                state["last_targeted_feature"] = selected_action
                state["exploit_pressure"] = max(0.0, float(state["exploit_pressure"]) - 0.10 * adaptation_strength)
            else:
                selected_action = "noop"
        if selected_action == "noop":
            state["clustering_bias"] = min(1.0, float(state["clustering_bias"]) + 0.08 * adaptation_strength)
            state["latency_noise_scale"] = max(0.0, float(state["latency_noise_scale"]) - 0.05 * adaptation_strength)
            state["honest_connection_bias"] = max(0.0, float(state["honest_connection_bias"]) - 0.07 * adaptation_strength)
            state["randomization"] = max(0.0, float(state["randomization"]) - 0.05 * adaptation_strength)
            state["exploit_pressure"] = min(1.0, float(state["exploit_pressure"]) + 0.12 * adaptation_strength)
        action_counts = dict(state.get("target_action_counts", {}))
        action_counts[selected_action] = int(action_counts.get(selected_action, 0)) + 1
        state["target_action_counts"] = action_counts

        if recent_profit >= HIGH_PROFIT_EXPLOIT_THRESHOLD:
            state["exploit_pressure"] = min(1.0, float(state["exploit_pressure"]) + 0.10)
        if perceived_score >= threshold or uncertainty >= DEFENSIVE_UNCERTAINTY_THRESHOLD:
            state["latency_noise_scale"] = min(3.0, float(state["latency_noise_scale"]) + 0.08)
            state["clustering_bias"] = max(0.02, float(state["clustering_bias"]) - 0.05)

        last_reset = int(state.get("last_identity_reset_timestep", NO_RESET_SENTINEL))
        reset_trigger = (
            perceived_score >= threshold * IDENTITY_RESET_HIGH_SCORE_FACTOR
            or recent_profit <= IDENTITY_RESET_LOW_REWARD_THRESHOLD
        )
        if reset_trigger and (timestep - last_reset) >= IDENTITY_RESET_COOLDOWN_TIMESTEPS:
            reset_count = int(state.get("identity_reset_count", 0)) + 1
            state["identity_reset_count"] = reset_count
            state["public_node_id"] = f"node-{node_id}-reset-{reset_count}"
            state["last_identity_reset_timestep"] = timestep
            state["detected_since_reset"] = perceived_score >= threshold
            state["stealth_budget"] = max_budget
            state["exploit_pressure"] = max(0.0, float(state["exploit_pressure"]) - 0.25)
            state["honest_connection_bias"] = min(1.0, float(state["honest_connection_bias"]) + 0.20)
            state["randomization"] = min(1.0, float(state["randomization"]) + 0.12)
        else:
            has_reset_before = int(state.get("last_identity_reset_timestep", NO_RESET_SENTINEL)) != NO_RESET_SENTINEL
            if has_reset_before and perceived_score >= threshold:
                state["post_reset_detections"] = int(state.get("post_reset_detections", 0)) + 1

        if strategy == "low_and_slow":
            state["exploit_pressure"] = min(float(state["exploit_pressure"]), 0.45)
            state["clustering_bias"] = min(float(state["clustering_bias"]), 0.50)
        elif strategy == "camouflage":
            state["honest_connection_bias"] = max(float(state["honest_connection_bias"]), 0.55)
        elif strategy == "burst_attack" and bool(state["burst_active"]) and score < threshold:
            state["exploit_pressure"] = min(1.0, float(state["exploit_pressure"]) + 0.25 + float(rng.uniform(0.0, 0.2)))
        elif strategy == "perfect_mimic":
            state["latency_noise_scale"] = max(0.05, float(state["latency_noise_scale"]) * 0.92)
            state["honest_connection_bias"] = min(1.0, max(float(state["honest_connection_bias"]), 0.72))
        elif strategy == "slow_drift":
            warmup = int(state.get("warmup_steps", 1))
            phase = max(0.0, (timestep - warmup) / max(total_timesteps - warmup, 1))
            state["drift_strength"] = phase
            state["exploit_pressure"] = float(np.clip(0.20 + 0.75 * phase, 0.0, 1.0))
        elif strategy == "decoy_attacker":
            if bool(state.get("is_decoy", False)):
                state["exploit_pressure"] = min(1.0, float(state["exploit_pressure"]) + 0.14)
                state["latency_noise_scale"] = min(3.0, float(state["latency_noise_scale"]) + 0.20)
            else:
                state["honest_connection_bias"] = min(1.0, float(state["honest_connection_bias"]) + 0.08)
        elif strategy == "mixed_cluster":
            state["honest_connection_bias"] = min(1.0, max(float(state["honest_connection_bias"]), 0.58))


def _update_attacker_economic_feedback(
    smart_states: Dict[int, Dict[str, object]],
    fraud_scores: Dict[int, float],
    threshold: float,
) -> None:
    for node_id, state in smart_states.items():
        detected = float(fraud_scores.get(node_id, 0.0)) >= threshold
        leakage = RESIDUAL_DETECTED_ATTACKER_LEAK * float(np.clip(state.get("exploit_pressure", 0.3), 0.0, 1.0))
        reward = (DETECTED_ATTACKER_REWARD + leakage) if detected else UNDETECTED_ATTACKER_REWARD
        reward_history = list(state.get("reward_history", []))
        reward_history.append(float(reward))
        state["reward_history"] = reward_history[-24:]
        state["latest_reward"] = float(reward)
        state["cumulative_reward"] = float(state.get("cumulative_reward", 0.0)) + float(reward)
        if reward >= (UNDETECTED_ATTACKER_REWARD - 0.3):
            state["exploit_pressure"] = min(1.0, float(state.get("exploit_pressure", 0.3)) + 0.06)
        if detected:
            state["latency_noise_scale"] = min(3.0, float(state.get("latency_noise_scale", 0.3)) + 0.10)
            state["honest_connection_bias"] = min(1.0, float(state.get("honest_connection_bias", 0.2)) + 0.08)


def build_network_simulation(
    total_nodes: int = 200,
    honest_ratio: float = 0.85,
    naive_attacker_ratio: float = 0.105,
    smart_attacker_ratio: float = 0.045,
    attacker_cluster_center: Coordinate = (1.3521, 103.8198),
    attacker_cluster_continent: str = "asia",
    honest_anomaly_rate: float = 0.12,
    honest_unstable_rate: float = 0.10,
    gray_zone_rate: float = 0.10,
    ambiguity_floor_rate: float = 0.04,
    measurement_error_std: float = 0.05,
    missing_latency_rate: float = 0.03,
    packet_loss_rate: float = 0.04,
    partial_visibility_rate: float = 0.06,
    delayed_observation_rate: float = 0.06,
    noise_spike_rate: float = 0.025,
    noise_level: float = 0.30,
    attacker_sophistication: float = 0.72,
    visibility_level: float = 0.74,
    label_noise_rate: float = 0.04,
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
    defender_retrain_interval: int = 4,
    defender_window_size: int = 6,
    difficulty_level: str = "medium",
) -> SimulationResult:
    if total_nodes < 10:
        raise ValueError("total_nodes must be at least 10 for a meaningful topology")
    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")

    ratio_sum = honest_ratio + naive_attacker_ratio + smart_attacker_ratio
    if not np.isclose(ratio_sum, 1.0, atol=RATIO_SUM_TOLERANCE):
        raise ValueError("Node-type ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    normalized_difficulty_level = str(difficulty_level).strip().lower()
    difficulty = _difficulty_profile(normalized_difficulty_level)
    noise_level = float(np.clip(noise_level, 0.0, 1.0))
    noise_level = float(np.clip(noise_level * float(difficulty["noise_multiplier"]), 0.0, 1.0))
    attacker_sophistication = float(np.clip(attacker_sophistication * float(difficulty["attacker_intelligence"]), 0.0, 1.0))
    visibility_level = float(np.clip(visibility_level * float(difficulty["visibility_multiplier"]), 0.0, 1.0))
    label_noise_rate = float(
        np.clip(label_noise_rate * float(difficulty["label_corruption_multiplier"]), 0.0, 0.25)
    )
    partial_label_rate = float(np.clip(difficulty["partial_label_rate"], 0.0, 0.75))
    delayed_label_steps = int(max(0, round(float(difficulty["delayed_label_steps"]))))
    suboptimal_action_rate = float(np.clip(difficulty["suboptimal_action_rate"], 0.0, 0.50))
    decision_epsilon = float(np.clip(difficulty["decision_epsilon"], 0.0, 0.50))
    feedback_noise_std = float(np.clip(difficulty["feedback_noise_std"], 0.0, 0.35))
    feedback_delay_steps = int(max(0, round(float(difficulty["feedback_delay_steps"]))))
    strategy_mutation_rate = float(np.clip(difficulty["strategy_mutation_rate"], 0.0, 0.40))
    budget_fraction = float(DEFENDER_BUDGET_FRACTION_BY_DIFFICULTY.get(normalized_difficulty_level, 0.07))
    defender_action_budget = int(max(3, round(total_nodes * budget_fraction)))

    measurement_error_std = float(measurement_error_std * (1.0 + 1.8 * noise_level))
    missing_latency_rate = float(np.clip(missing_latency_rate + 0.10 * noise_level + 0.08 * (1.0 - visibility_level), 0.0, 0.45))
    packet_loss_rate = float(np.clip(packet_loss_rate + 0.08 * noise_level + 0.05 * (1.0 - visibility_level), 0.0, 0.45))
    delayed_observation_rate = float(np.clip(delayed_observation_rate + 0.12 * noise_level, 0.0, 0.55))
    partial_visibility_rate = float(np.clip(partial_visibility_rate + 0.16 * (1.0 - visibility_level) + 0.08 * noise_level, 0.0, 0.60))
    noise_spike_rate = float(np.clip(noise_spike_rate + 0.08 * noise_level, 0.0, 0.35))
    honest_anomaly_rate = float(np.clip(honest_anomaly_rate + 0.14 * noise_level, 0.0, 0.55))
    gray_zone_rate = float(np.clip(gray_zone_rate + 0.10 * noise_level, 0.0, 0.55))
    ambiguity_floor_rate = float(np.clip(ambiguity_floor_rate + 0.08 * noise_level, 0.0, 0.40))

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
    effective_strategy_mix = _compose_smart_strategy_mix(
        strategy_mix=smart_strategy_mix,
        attacker_sophistication=attacker_sophistication,
    )
    smart_strategies = _sample_smart_attacker_strategies(smart_ids, rng, strategy_mix=effective_strategy_mix)
    high_value_regions = ["new_york", "tokyo", "london", "singapore"]
    high_value_region = str(high_value_regions[int(rng.integers(0, len(high_value_regions)))])

    nodes = (
        _sample_honest_nodes(
            honest_ids,
            rng,
            anomaly_rate=honest_anomaly_rate,
            unstable_rate=honest_unstable_rate,
            gray_zone_rate=gray_zone_rate,
            ambiguity_floor_rate=ambiguity_floor_rate,
        )
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
    smart_states = _initialize_smart_states(
        nodes,
        total_timesteps=time_steps,
        rng=rng,
        feature_priority=feature_priority,
        attacker_sophistication=attacker_sophistication,
    )
    campaigns = _initialize_attack_campaigns(smart_states=smart_states, rng=rng)
    learning_state = StrategyLearningState(strategy_mix=effective_strategy_mix, temperature=0.28)
    attacker_policy = StrategicAttackerPolicy()
    defender_policy = AdaptiveDefenderPolicy(
        base_threshold=fraud_score_threshold,
        retrain_interval=defender_retrain_interval,
        sliding_window_steps=defender_window_size,
        dynamic_threshold=True,
        quarantine_duration=2,
        action_budget_per_step=defender_action_budget,
        threshold_exploration_rate=float(np.clip(0.16 - 0.10 * attacker_sophistication, 0.03, 0.20)),
        random_state=seed + 17,
    )
    defender_policy.register_nodes([node.node_id for node in nodes])
    scenario_state: Dict[str, Any] = {
        "high_value_region": high_value_region,
        "regional_attack_active": False,
        "regional_target_region": high_value_region,
        "sybil_swarm_nodes": set(),
        "corrupted_honest_nodes": set(),
    }

    snapshots: List[TimeStepSnapshot] = []
    fraud_scores_over_time: List[Dict[int, float]] = []
    uncertainty_over_time: List[Dict[int, float]] = []
    scenario_events: List[AttackScenarioEvent] = []
    lag_history: List[np.ndarray] = []
    delayed_scores: Dict[int, float] = {}
    strategy_dominance_over_time: List[float] = []

    for timestep in range(time_steps):
        defender_policy.prepare_timestep(timestep)
        if enable_system_attacks:
            scenario_events.extend(_activate_attack_scenarios(timestep=timestep, nodes=nodes, scenario_state=scenario_state, rng=rng))
        quarantined_nodes = {node_id for node_id in smart_states if defender_policy.is_quarantined(node_id, timestep)}
        latency_matrix = _build_latency_matrix(
            nodes=nodes,
            timestep=timestep,
            total_timesteps=time_steps,
            rng=rng,
            measurement_error_std=measurement_error_std,
            packet_loss_rate=packet_loss_rate,
            missing_latency_rate=missing_latency_rate,
            delayed_observation_rate=delayed_observation_rate,
            noise_spike_rate=noise_spike_rate,
            smart_states=smart_states,
            scenario_state=scenario_state,
            lag_reference_matrices=lag_history[-MAX_OBSERVATION_LAG:],
        )
        lag_history.append(latency_matrix.copy())
        if len(lag_history) > (MAX_OBSERVATION_LAG + 1):
            lag_history.pop(0)
        peer_graph = _build_peer_graph(
            nodes=nodes,
            latency_matrix=latency_matrix,
            timestep=timestep,
            total_timesteps=time_steps,
            partial_visibility_rate=partial_visibility_rate,
            rng=rng,
            smart_states=smart_states,
            scenario_state=scenario_state,
            quarantined_nodes=quarantined_nodes,
        )

        fraud_scores: Dict[int, float] = {}
        uncertainty_scores: Dict[int, float] = {}
        model_df = pd.DataFrame()
        if model_for_adaptation is not None and feature_columns:
            fraud_scores, uncertainty_scores, model_df = _score_snapshot_for_adaptation(
                nodes=nodes,
                latency_matrix=latency_matrix,
                graph=peer_graph,
                timestep=timestep,
                model=model_for_adaptation,
                feature_columns=feature_columns,
            )
            blended_scores: Dict[int, float] = {}
            for node_id, score in fraud_scores.items():
                previous = float(delayed_scores.get(node_id, score))
                blended = DELAYED_SIGNAL_MIX * previous + (1.0 - DELAYED_SIGNAL_MIX) * float(score)
                blended_scores[int(node_id)] = float(np.clip(blended, 0.0, 1.0))
            fraud_scores = blended_scores
            delayed_scores = dict(blended_scores)
            node_region_map = {node.node_id: node.region for node in nodes}
            node_label_map = {node.node_id: node.label for node in nodes}
            prioritized_node_ids = defender_policy.prioritize_nodes_for_investigation(
                fraud_scores=fraud_scores,
                uncertainty_scores=uncertainty_scores,
                node_regions=node_region_map,
            )
            effective_pred = []
            y_true = []
            for node_id in prioritized_node_ids:
                score = float(fraud_scores.get(node_id, 0.0))
                uncertainty = float(uncertainty_scores.get(node_id, 0.0))
                label = node_label_map.get(node_id, "honest")
                region = node_region_map.get(node_id, "unknown")
                flagged = defender_policy.assess_node(
                    node_id=node_id,
                    label=label,
                    region=region,
                    fraud_score=float(score),
                    uncertainty=uncertainty,
                    timestep=timestep,
                )
                y_true.append(0 if label == "honest" else 1)
                effective_pred.append(1 if flagged else 0)
            defender_policy.training_history.append(model_df.copy())
            defender_policy.timestep_metrics(
                y_true=np.array(y_true, dtype=int),
                y_pred=np.array(effective_pred, dtype=int),
                threshold=defender_policy.current_threshold,
            )
            retrained_bundle = defender_policy.maybe_retrain(
                timestep=timestep,
                feature_columns=feature_columns,
                random_state=seed,
            )
            if retrained_bundle is not None:
                model_for_adaptation = retrained_bundle
        fraud_scores_over_time.append(fraud_scores)
        uncertainty_over_time.append(uncertainty_scores)

        snapshots.append(
            TimeStepSnapshot(
                timestep=timestep,
                latency_matrix=latency_matrix,
                peer_graph=peer_graph,
                fraud_scores=fraud_scores,
                uncertainty_scores=uncertainty_scores,
            )
        )
        _update_attacker_economic_feedback(
            smart_states=smart_states,
            fraud_scores=fraud_scores,
            threshold=defender_policy.current_threshold,
        )

        if smart_states and timestep < time_steps - 1:
            _update_smart_attacker_states(
                smart_states=smart_states,
                campaigns=campaigns,
                fraud_scores=fraud_scores,
                uncertainty_scores=uncertainty_scores,
                model_df=model_df,
                model_for_adaptation=model_for_adaptation if model_for_adaptation is not None else {},
                feature_columns=feature_columns,
                attacker_policy=attacker_policy,
                threshold=defender_policy.current_threshold,
                timestep=timestep,
                total_timesteps=time_steps,
                adaptation_base=adaptation_base,
                adaptation_growth=adaptation_growth,
                suboptimal_action_rate=suboptimal_action_rate,
                decision_epsilon=decision_epsilon,
                feedback_noise_std=feedback_noise_std,
                feedback_delay_steps=feedback_delay_steps,
                rng=rng,
            )
            strategy_rewards = _strategy_rewards_from_states(
                smart_states=smart_states,
                threshold=defender_policy.current_threshold,
            )
            learning_state = update_learning_state(learning_state, strategy_rewards=strategy_rewards)
            for state in smart_states.values():
                if rng.random() < strategy_mutation_rate:
                    new_strategy = str(rng.choice(SMART_ATTACKER_STRATEGY_ARRAY))
                    state["strategy"] = new_strategy
                    state["is_emergent_strategy"] = True
                if rng.random() < STRATEGY_RESAMPLING_RATE:
                    sampled_strategy = str(
                        rng.choice(
                            SMART_ATTACKER_STRATEGY_ARRAY,
                            p=np.array(
                                [learning_state.strategy_mix.get(name, 0.0) for name in SMART_ATTACKER_STRATEGIES],
                                dtype=float,
                            ),
                        )
                    )
                    state["strategy"] = sampled_strategy
            effective_strategy_mix = evolve_strategy_mix(
                current_mix=learning_state.strategy_mix,
                mutation_rate=strategy_mutation_rate,
                rng=rng,
            )
            learning_state.strategy_mix = dict(effective_strategy_mix)
            strategy_dominance_over_time.append(
                float(max(learning_state.strategy_mix.values()) if learning_state.strategy_mix else 0.0)
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
        total_resets = int(sum(int(state.get("identity_reset_count", 0)) for state in smart_states.values()))
        total_post_reset_detections = int(sum(int(state.get("post_reset_detections", 0)) for state in smart_states.values()))
        avg_reward = float(np.mean([float(state.get("cumulative_reward", 0.0)) for state in smart_states.values()]))
    else:
        avg_remaining_budget = 0.0
        total_resets = 0
        total_post_reset_detections = 0
        avg_reward = 0.0
    warmup_values = [int(state.get("warmup_steps", 1)) for state in smart_states.values()]
    defender_summary = summarize_defender_metrics(defender_policy, current_timestep=time_steps - 1)
    scenario_metrics: Dict[str, float] = {
        **event_counts,
        **defender_summary,
        "scenario_event_count": float(len(scenario_events)),
        "high_value_region_node_count": float(sum(1 for n in nodes if n.region == high_value_region)),
        "corrupted_honest_node_count": float(len(scenario_state.get("corrupted_honest_nodes", set()))),
        "avg_smart_attacker_remaining_stealth_budget": avg_remaining_budget,
        "identity_reset_frequency": float(total_resets / max(len(smart_states), 1)),
        "post_reset_detection_count": float(total_post_reset_detections),
        "avg_smart_attacker_cumulative_reward": avg_reward,
        "gray_zone_node_count": float(sum(1 for n in nodes if n.gray_zone)),
        "ambiguity_anchor_count": float(sum(1 for n in nodes if n.ambiguity_anchor)),
        "avg_attacker_warmup_steps": float(np.mean(warmup_values) if warmup_values else 0.0),
        "noise_level": noise_level,
        "attacker_sophistication": attacker_sophistication,
        "visibility_level": visibility_level,
        "label_noise_rate": label_noise_rate,
        "difficulty_level": float(Difficulties.index(normalized_difficulty_level)),
        "partial_label_rate": partial_label_rate,
        "delayed_label_steps": float(delayed_label_steps),
        "suboptimal_action_rate": suboptimal_action_rate,
        "decision_epsilon": decision_epsilon,
        "feedback_noise_std": feedback_noise_std,
        "feedback_delay_steps": float(feedback_delay_steps),
        "strategy_mutation_rate": strategy_mutation_rate,
        "strategy_dominance_score": float(max(learning_state.strategy_mix.values()) if learning_state.strategy_mix else 0.0),
        "strategy_dominance_over_time": float(np.mean(strategy_dominance_over_time) if strategy_dominance_over_time else 0.0),
        "campaign_count": float(len(campaigns)),
        "leader_count": float(sum(1 for state in smart_states.values() if state.get("campaign_role") == "leader")),
        "decoy_count": float(sum(1 for state in smart_states.values() if state.get("campaign_role") == "decoy")),
        "executor_count": float(sum(1 for state in smart_states.values() if state.get("campaign_role") == "executor")),
        "measurement_error_std_effective": measurement_error_std,
        "missing_latency_rate_effective": missing_latency_rate,
        "packet_loss_rate_effective": packet_loss_rate,
        "partial_visibility_rate_effective": partial_visibility_rate,
        "seed": float(seed),
    }
    for strategy_name, weight in learning_state.strategy_mix.items():
        scenario_metrics[f"strategy_mix_{strategy_name}"] = float(weight)
    return SimulationResult(
        nodes=nodes,
        latency_matrix=final_snapshot.latency_matrix,
        peer_graph=final_snapshot.peer_graph,
        time_steps=snapshots,
        fraud_scores_over_time=fraud_scores_over_time,
        uncertainty_over_time=uncertainty_over_time,
        scenario_events=scenario_events,
        scenario_metrics=scenario_metrics,
    )
