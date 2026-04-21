from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd


@dataclass(frozen=True)
class DetectionMetricsResult:
    per_node: pd.DataFrame
    summary: Dict[str, float]
    false_positives_over_time: pd.DataFrame


def compute_detection_metrics(
    fraud_scores_over_time: List[Dict[int, float]],
    node_labels: Dict[int, str],
    threshold: float,
    within_steps: int = 5,
    node_regions: Optional[Dict[int, str]] = None,
    high_value_regions: Optional[Set[str]] = None,
) -> DetectionMetricsResult:
    if not fraud_scores_over_time:
        empty = pd.DataFrame(
            columns=[
                "node_id",
                "label_name",
                "first_detection_timestep",
                "detection_consistency",
                "detection_delay",
                "was_detected",
            ]
        )
        return DetectionMetricsResult(per_node=empty, summary={}, false_positives_over_time=pd.DataFrame())

    all_node_ids: Iterable[int] = node_labels.keys()
    timestep_count = len(fraud_scores_over_time)
    node_regions = node_regions or {node_id: "unknown" for node_id in node_labels}
    high_value_regions = high_value_regions or set()
    node_rows = []
    fp_rows = []
    detected_attackers_within_n = 0
    detected_attackers_within_2 = 0
    total_attackers = sum(1 for label in node_labels.values() if label != "honest")
    detection_delays = []

    for timestep, score_map in enumerate(fraud_scores_over_time):
        fp_count = 0
        for node_id, score in score_map.items():
            if node_labels.get(node_id, "honest") == "honest" and score >= threshold:
                fp_count += 1
        fp_rows.append({"timestep": timestep, "false_positives": fp_count})

    for node_id in all_node_ids:
        label_name = node_labels[node_id]
        flagged_timesteps = []
        for timestep, score_map in enumerate(fraud_scores_over_time):
            if float(score_map.get(node_id, 0.0)) >= threshold:
                flagged_timesteps.append(timestep)

        first_detection = flagged_timesteps[0] if flagged_timesteps else -1
        detection_consistency = float(len(flagged_timesteps) / max(timestep_count, 1))
        detection_delay = first_detection if first_detection >= 0 else timestep_count
        was_detected = first_detection >= 0

        if label_name != "honest":
            detection_delays.append(float(detection_delay))
            if was_detected and first_detection <= within_steps:
                detected_attackers_within_n += 1
            if was_detected and first_detection <= 2:
                detected_attackers_within_2 += 1

        node_rows.append(
            {
                "node_id": int(node_id),
                "label_name": label_name,
                "first_detection_timestep": int(first_detection),
                "detection_consistency": detection_consistency,
                "detection_delay": float(detection_delay),
                "was_detected": bool(was_detected),
            }
        )

    per_node = pd.DataFrame(node_rows).sort_values("node_id").reset_index(drop=True)
    false_positives_over_time = pd.DataFrame(fp_rows).sort_values("timestep").reset_index(drop=True)
    avg_detection_delay = float(sum(detection_delays) / max(len(detection_delays), 1))
    pct_detected_within_n = float(100.0 * detected_attackers_within_n / max(total_attackers, 1))

    summary = {
        "avg_detection_delay": avg_detection_delay,
        "median_detection_delay": float(pd.Series(detection_delays).median() if detection_delays else 0.0),
        f"pct_attackers_detected_within_{within_steps}_steps": pct_detected_within_n,
        "early_detection_rate_within_2_steps": float(100.0 * detected_attackers_within_2 / max(total_attackers, 1)),
        "avg_false_positives_over_time": float(false_positives_over_time["false_positives"].mean()),
    }
    if high_value_regions:
        attacker_ids = [node_id for node_id, label in node_labels.items() if label != "honest"]
        high_value_attackers = [nid for nid in attacker_ids if node_regions.get(nid, "") in high_value_regions]
        high_value_detections = 0
        for node_id in high_value_attackers:
            flagged = any(float(scores.get(node_id, 0.0)) >= threshold for scores in fraud_scores_over_time)
            if flagged:
                high_value_detections += 1
        summary["high_value_attacker_detection_rate"] = float(high_value_detections / max(len(high_value_attackers), 1))
        if high_value_attackers:
            region_counts: Dict[str, int] = {}
            for node_id in high_value_attackers:
                region = str(node_regions.get(node_id, "unknown"))
                region_counts[region] = region_counts.get(region, 0) + 1
            summary["fraud_concentration_by_region"] = float(max(region_counts.values()) / max(len(high_value_attackers), 1))
        else:
            summary["fraud_concentration_by_region"] = 0.0
    return DetectionMetricsResult(
        per_node=per_node,
        summary=summary,
        false_positives_over_time=false_positives_over_time,
    )


def compute_system_cost_metrics(
    per_node: pd.DataFrame,
    false_positives_over_time: pd.DataFrame,
    cost_of_false_positive: float = 1.0,
    cost_of_false_negative: float = 8.0,
    detection_latency_penalty: float = 2.5,
    economic_loss_due_to_delay: float = 1.8,
) -> Dict[str, float]:
    """Compute product-oriented system cost metrics for model selection and reporting."""
    if per_node.empty:
        return {
            "cost_of_false_positive": 0.0,
            "cost_of_false_negative": 0.0,
            "detection_latency_penalty": 0.0,
            "economic_loss_due_to_delay": 0.0,
            "total_system_cost": 0.0,
        }
    honest_mask = per_node["label_name"] == "honest"
    attacker_mask = ~honest_mask
    false_positive_count = int((honest_mask & per_node["was_detected"].astype(bool)).sum())
    false_negative_count = int((attacker_mask & (~per_node["was_detected"].astype(bool))).sum())
    avg_delay = float(pd.to_numeric(per_node.loc[attacker_mask, "detection_delay"], errors="coerce").fillna(0.0).mean())
    fp_cost = float(cost_of_false_positive * false_positive_count)
    fn_cost = float(cost_of_false_negative * false_negative_count)
    latency_cost = float(detection_latency_penalty * avg_delay)
    delayed_economic_loss = float(economic_loss_due_to_delay * avg_delay * max(int(attacker_mask.sum()), 1))
    total = fp_cost + fn_cost + latency_cost + delayed_economic_loss
    return {
        "cost_of_false_positive": fp_cost,
        "cost_of_false_negative": fn_cost,
        "detection_latency_penalty": latency_cost,
        "economic_loss_due_to_delay": delayed_economic_loss,
        "avg_false_positives_over_time": float(false_positives_over_time["false_positives"].mean())
        if not false_positives_over_time.empty
        else 0.0,
        "total_system_cost": float(total),
    }
