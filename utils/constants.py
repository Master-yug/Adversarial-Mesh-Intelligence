BASE_FEATURE_COLUMNS = [
    "rtt_variance",
    "avg_latency_to_peers",
    "claimed_inferred_distance_mismatch",
    "unique_peers",
    "clustering_coefficient",
    "reciprocity_score",
    "latency_skewness",
    "latency_kurtosis",
    "peer_geographic_diversity",
    "latency_inconsistency_score",
    "neighbor_trust_score",
    "edge_asymmetry",
    "max_latency_spike",
    "latency_trend_slope",
    "behavior_volatility",
    "sudden_change_score",
    "burst_activity_score",
]

AGGREGATION_SUFFIXES = ("mean", "std", "last")

FEATURE_COLUMNS = [
    f"{base}_{suffix}" for base in BASE_FEATURE_COLUMNS for suffix in AGGREGATION_SUFFIXES
]

FEATURE_MEANINGS = {
    "rtt_variance": "Variance of observed outbound peer latencies.",
    "avg_latency_to_peers": "Average observed outbound peer latency.",
    "claimed_inferred_distance_mismatch": "Distance between node claimed location and peer-latency-inferred location.",
    "unique_peers": "Count of unique in/out peers visible for the node.",
    "clustering_coefficient": "Local clustering coefficient in the observed peer graph.",
    "reciprocity_score": "Fraction of outbound links that are reciprocated.",
    "latency_skewness": "Skewness of observed outbound latency distribution.",
    "latency_kurtosis": "Kurtosis of observed outbound latency distribution.",
    "peer_geographic_diversity": "Geographic spread of peer claimed locations.",
    "latency_inconsistency_score": "Mismatch between expected latency from claimed geography and observed latency.",
    "neighbor_trust_score": "Heuristic trust score of neighbors from graph and latency behavior.",
    "edge_asymmetry": "Average asymmetry between A→B and B→A observed latencies.",
    "max_latency_spike": "Maximum observed latency spike across timesteps for the node.",
    "latency_trend_slope": "Linear trend slope of average latency over time.",
    "behavior_volatility": "Temporal volatility of core behavioral features.",
    "sudden_change_score": "Magnitude of sudden changes between consecutive timesteps.",
    "burst_activity_score": "Fraction of timesteps with burst-like behavior spikes.",
}
