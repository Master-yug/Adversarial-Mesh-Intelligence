from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_CONNECTIVITY_MULTIPLIER = 1.0
CONNECTIVITY_REWARD_FACTOR = 0.5
REWARD_FLUCTUATION_AMPLITUDE = 0.16


@dataclass(frozen=True)
class EconomicSimulationResult:
    per_node_rewards: pd.DataFrame
    summary: Dict[str, float]


def _compute_dynamic_reward_multiplier(
    region_multiplier: float,
    connectivity_score: float,
    timestep: int,
    timestep_count: int,
) -> float:
    connectivity_multiplier = BASE_CONNECTIVITY_MULTIPLIER + CONNECTIVITY_REWARD_FACTOR * connectivity_score
    fluctuation = 1.0 + REWARD_FLUCTUATION_AMPLITUDE * np.sin((2.0 * np.pi * timestep) / max(timestep_count, 1))
    return float(region_multiplier * connectivity_multiplier * fluctuation)


def simulate_economic_rewards(
    fraud_scores_over_time: List[Dict[int, float]],
    node_labels: Dict[int, str],
    threshold: float,
    node_regions: Optional[Dict[int, str]] = None,
    connectivity_over_time: Optional[List[Dict[int, float]]] = None,
    high_value_regions: Optional[Dict[str, float]] = None,
    honest_reward: float = 1.0,
    attacker_reward_undetected: float = 1.8,
    attacker_reward_detected: float = 0.35,
    naive_reward_multiplier: float = 1.0,
    smart_reward_multiplier: float = 1.15,
    residual_leak_floor: float = 0.18,
    defense_fatigue: float = 0.28,
) -> EconomicSimulationResult:
    timestep_count = len(fraud_scores_over_time)
    node_regions = node_regions or {node_id: "unknown" for node_id in node_labels}
    high_value_regions = high_value_regions or {"new_york": 1.4, "tokyo": 1.5, "london": 1.35, "singapore": 1.45}
    connectivity_over_time = connectivity_over_time or [{node_id: 0.0 for node_id in node_labels} for _ in range(timestep_count)]
    reward_rows = []
    high_value_regions_set = set(high_value_regions.keys())

    for node_id, label in node_labels.items():
        total_reward = 0.0
        detected = False
        attacker_reward_without_detection = 0.0
        region = str(node_regions.get(node_id, "unknown"))
        region_multiplier = float(high_value_regions.get(region, 1.0))
        attack_attempts_high_value = 0
        attack_success_high_value = 0
        for timestep in range(timestep_count):
            score = float(fraud_scores_over_time[timestep].get(node_id, 0.0))
            if score >= threshold:
                detected = True

            connectivity_score = float(connectivity_over_time[timestep].get(node_id, 0.0))
            dynamic_multiplier = _compute_dynamic_reward_multiplier(
                region_multiplier=region_multiplier,
                connectivity_score=connectivity_score,
                timestep=timestep,
                timestep_count=timestep_count,
            )

            if label == "honest":
                reward = honest_reward * dynamic_multiplier
            else:
                attacker_base = attacker_reward_undetected
                if label == "smart_attacker":
                    attacker_base *= smart_reward_multiplier
                elif label == "naive_attacker":
                    attacker_base *= naive_reward_multiplier
                attacker_base *= dynamic_multiplier
                attacker_reward_without_detection += attacker_base
                fatigue_ratio = timestep / max(timestep_count - 1, 1)
                effective_detected_reward = (
                    attacker_reward_detected + residual_leak_floor + defense_fatigue * fatigue_ratio
                ) * dynamic_multiplier
                reward = effective_detected_reward if detected else attacker_base
                if region in high_value_regions_set:
                    attack_attempts_high_value += 1
                    if not detected:
                        attack_success_high_value += 1
            total_reward += reward

        reward_rows.append(
            {
                "node_id": int(node_id),
                "label_name": label,
                "region": region,
                "total_reward": float(total_reward),
                "attacker_reward_without_detection": float(attacker_reward_without_detection),
                "high_value_attack_attempts": int(attack_attempts_high_value),
                "high_value_attack_successes": int(attack_success_high_value),
            }
        )

    per_node_rewards = pd.DataFrame(reward_rows).sort_values("node_id").reset_index(drop=True)
    attacker_mask = per_node_rewards["label_name"] != "honest"
    honest_mask = per_node_rewards["label_name"] == "honest"

    total_attacker_reward = float(per_node_rewards.loc[attacker_mask, "total_reward"].sum())
    total_honest_reward = float(per_node_rewards.loc[honest_mask, "total_reward"].sum())
    baseline_attacker_reward = float(per_node_rewards.loc[attacker_mask, "attacker_reward_without_detection"].sum())
    fraud_reduction_after_detection = float(max(baseline_attacker_reward - total_attacker_reward, 0.0))
    total_system_reward = float(per_node_rewards["total_reward"].sum())
    total_fraud_profit = total_attacker_reward
    pct_reward_lost_to_fraud = float(100.0 * total_attacker_reward / max(total_system_reward, 1e-9))
    high_value_attacker_rows = per_node_rewards.loc[
        attacker_mask & per_node_rewards["region"].isin(list(high_value_regions_set))
    ]
    high_value_attempts = int(high_value_attacker_rows["high_value_attack_attempts"].sum())
    high_value_successes = int(high_value_attacker_rows["high_value_attack_successes"].sum())
    high_value_attack_success_rate = float(high_value_successes / max(high_value_attempts, 1))
    per_region_fraud = (
        per_node_rewards.loc[attacker_mask]
        .groupby("region", dropna=False)["total_reward"]
        .sum()
        .rename("fraud_reward")
        .reset_index()
    )
    total_region_fraud = float(per_region_fraud["fraud_reward"].sum()) if not per_region_fraud.empty else 0.0
    if total_region_fraud > 0.0:
        per_region_fraud["fraud_share"] = per_region_fraud["fraud_reward"] / total_region_fraud
    else:
        per_region_fraud["fraud_share"] = 0.0
    max_region_share = float(per_region_fraud["fraud_share"].max()) if not per_region_fraud.empty else 0.0

    summary = {
        "total_fraud_profit": total_fraud_profit,
        "fraud_reduction_after_detection": fraud_reduction_after_detection,
        "attacker_profit_vs_honest_profit_ratio": float(total_attacker_reward / max(total_honest_reward, 1e-9)),
        "pct_reward_lost_to_fraud": pct_reward_lost_to_fraud,
        "fraud_concentration_by_region": max_region_share,
        "high_value_attack_success_rate": high_value_attack_success_rate,
    }
    return EconomicSimulationResult(per_node_rewards=per_node_rewards, summary=summary)
