from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping
import numpy as np

from defender_engine.policies import tune_threshold

UNCERTAINTY_PENALTY_WEIGHT = 0.15
DELAYED_DETECTION_SIGNAL_WEIGHT = 0.5
MIN_THRESHOLD = 0.2
MAX_THRESHOLD = 0.95
SELECTION_STRATEGIES = ("risk_only", "risk_adjusted", "uncertainty_first")
ATTACKER_AGGRESSION_PRIOR = {
    "low_and_slow": 0.35,
    "burst_attack": 0.92,
    "camouflage": 0.58,
    "perfect_mimic": 0.62,
    "slow_drift": 0.48,
    "decoy_attacker": 0.66,
    "mixed_cluster": 0.74,
}
DEFAULT_BUDGET_GRID = (0.03, 0.05, 0.07, 0.10, 0.14, 0.18)

@dataclass
class DefenderAgent:
    threshold: float = 0.5
    budget: int | None = None
    budget_ratio: float = 0.07
    selection_strategy: str = "risk_adjusted"
    min_budget: int = 1
    fp_cost: float = 1.0
    fn_cost: float = 8.0
    delayed_detection_cost: float = 2.5
    investigation_cost: float = 0.7
    delayed_detection_penalty: float = 1.8
    threshold_step: float = 0.03
    threshold_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.selection_strategy not in SELECTION_STRATEGIES:
            self.selection_strategy = "risk_adjusted"

    def _effective_budget(self, candidate_count: int, budget_ratio_override: float | None = None) -> int:
        if self.budget is not None:
            return max(int(self.budget), 0)
        ratio = float(self.budget_ratio if budget_ratio_override is None else budget_ratio_override)
        ratio_budget = int(round(float(np.clip(ratio, 0.0, 1.0)) * int(candidate_count)))
        return max(int(self.min_budget), ratio_budget)

    def _candidate_rankings(
        self,
        fraud_scores: Mapping[int, float],
        candidates: Iterable[int],
        uncertainty_scores: Mapping[int, float] | None = None,
        selection_strategy: str | None = None,
    ) -> List[int]:
        uncertainty_scores = uncertainty_scores or {}
        strategy = str(selection_strategy or self.selection_strategy)
        if strategy not in SELECTION_STRATEGIES:
            strategy = "risk_adjusted"

        def rank_score(node_id: int) -> float:
            risk = float(fraud_scores.get(int(node_id), 0.0))
            uncertainty = float(uncertainty_scores.get(int(node_id), 0.0))
            if strategy == "risk_only":
                return risk
            if strategy == "uncertainty_first":
                return risk + 0.20 * uncertainty
            return risk - UNCERTAINTY_PENALTY_WEIGHT * uncertainty

        return sorted((int(node_id) for node_id in candidates), key=rank_score, reverse=True)

    def select_nodes(
        self,
        fraud_scores: Mapping[int, float],
        node_selection: Iterable[int] | None = None,
        uncertainty_scores: Mapping[int, float] | None = None,
    ) -> List[int]:
        candidates = list(node_selection) if node_selection is not None else list(fraud_scores.keys())
        ranked = self._candidate_rankings(
            fraud_scores=fraud_scores,
            candidates=candidates,
            uncertainty_scores=uncertainty_scores,
            selection_strategy=self.selection_strategy,
        )
        selected = [int(node_id) for node_id in ranked if float(fraud_scores.get(int(node_id), 0.0)) >= float(self.threshold)]
        return selected[: self._effective_budget(candidate_count=len(candidates))]

    def estimate_total_cost(
        self,
        false_positive_rate: float,
        false_negative_rate: float,
        delayed_detection_rate: float = 0.0,
        investigation_rate: float = 0.0,
    ) -> float:
        fp_component = self.fp_cost * float(false_positive_rate)
        fn_component = self.fn_cost * float(false_negative_rate)
        delayed_component = self.delayed_detection_cost * float(delayed_detection_rate)
        investigation_component = self.investigation_cost * float(investigation_rate)
        delay_penalty_component = self.delayed_detection_penalty * float(delayed_detection_rate)
        return float(fp_component + fn_component + delayed_component + investigation_component + delay_penalty_component)

    def optimize_threshold(
        self,
        fraud_scores: Mapping[int, float],
        labels: Mapping[int, int],
        predicted_attacker_pressure: float = 0.0,
        grid_size: int = 15,
    ) -> Dict[str, float]:
        optimized = self.optimize_policy(
            fraud_scores=fraud_scores,
            labels=labels,
            uncertainty_scores=None,
            attacker_strategy_estimate={"_pressure_proxy": float(np.clip(predicted_attacker_pressure, 0.0, 1.0))},
            grid_size=grid_size,
        )
        return {
            "threshold": float(optimized["threshold"]),
            "total_cost": float(optimized["total_cost"]),
            "false_positive_rate": float(optimized["false_positive_rate"]),
            "false_negative_rate": float(optimized["false_negative_rate"]),
            "investigation_rate": float(optimized["investigation_rate"]),
        }

    def optimize_policy(
        self,
        fraud_scores: Mapping[int, float],
        labels: Mapping[int, int],
        uncertainty_scores: Mapping[int, float] | None = None,
        attacker_strategy_estimate: Mapping[str, float] | None = None,
        grid_size: int = 15,
        budget_grid: Iterable[float] = DEFAULT_BUDGET_GRID,
    ) -> Dict[str, float | str]:
        if not fraud_scores:
            return {
                "threshold": float(self.threshold),
                "budget_ratio": float(self.budget_ratio),
                "selection_strategy": str(self.selection_strategy),
                "total_cost": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "investigation_rate": 0.0,
            }
        candidates = np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, max(int(grid_size), 3))
        node_ids = [int(node_id) for node_id in fraud_scores.keys()]
        budget_candidates = sorted(
            {
                float(np.clip(v, 0.0, 1.0))
                for v in list(budget_grid) + [float(self.budget_ratio)]
            }
        )

        pressure_proxy = 0.0
        estimate = attacker_strategy_estimate or {}
        if "_pressure_proxy" in estimate:
            pressure_proxy = float(np.clip(float(estimate.get("_pressure_proxy", 0.0)), 0.0, 1.0))
        else:
            pressure_proxy = float(
                np.clip(
                    sum(
                        float(prob) * float(ATTACKER_AGGRESSION_PRIOR.get(name, 0.5))
                        for name, prob in estimate.items()
                    ),
                    0.0,
                    1.0,
                )
            )
        best_threshold = float(self.threshold)
        best_budget_ratio = float(self.budget_ratio)
        best_selection_strategy = str(self.selection_strategy)
        best_cost = float("inf")
        best_fp_rate = 0.0
        best_fn_rate = 0.0
        best_investigation_rate = 0.0
        for candidate in candidates:
            for budget_ratio in budget_candidates:
                budget = self._effective_budget(candidate_count=len(node_ids), budget_ratio_override=budget_ratio)
                for selection_strategy in SELECTION_STRATEGIES:
                    ranked = self._candidate_rankings(
                        fraud_scores=fraud_scores,
                        candidates=node_ids,
                        uncertainty_scores=uncertainty_scores,
                        selection_strategy=selection_strategy,
                    )
                    selected = [n for n in ranked if float(fraud_scores.get(n, 0.0)) >= float(candidate)][:budget]
                    selected_set = set(int(v) for v in selected)
                    attacker_ids = [n for n in node_ids if int(labels.get(n, 0)) == 1]
                    honest_ids = [n for n in node_ids if int(labels.get(n, 0)) == 0]
                    fp = sum(1 for n in honest_ids if n in selected_set)
                    fn = sum(1 for n in attacker_ids if n not in selected_set)
                    fp_rate = float(fp / max(len(honest_ids), 1))
                    fn_rate = float(fn / max(len(attacker_ids), 1))
                    investigation_rate = float(len(selected_set) / max(len(node_ids), 1))
                    delayed_rate = float(np.clip(pressure_proxy * fn_rate, 0.0, 1.0))
                    total_cost = self.estimate_total_cost(
                        false_positive_rate=fp_rate,
                        false_negative_rate=fn_rate,
                        delayed_detection_rate=delayed_rate,
                        investigation_rate=investigation_rate,
                    )
                    if total_cost < best_cost:
                        best_threshold = float(candidate)
                        best_budget_ratio = float(budget_ratio)
                        best_selection_strategy = str(selection_strategy)
                        best_cost = float(total_cost)
                        best_fp_rate = fp_rate
                        best_fn_rate = fn_rate
                        best_investigation_rate = investigation_rate
        self.threshold = float(best_threshold)
        self.budget_ratio = float(np.clip(best_budget_ratio, 0.0, 1.0))
        self.selection_strategy = best_selection_strategy
        self.threshold_history.append(float(self.threshold))
        self.cost_history.append(float(best_cost))
        return {
            "threshold": float(best_threshold),
            "budget_ratio": float(self.budget_ratio),
            "selection_strategy": str(self.selection_strategy),
            "total_cost": float(best_cost),
            "false_positive_rate": float(best_fp_rate),
            "false_negative_rate": float(best_fn_rate),
            "investigation_rate": float(best_investigation_rate),
        }

    def update_policy(
        self,
        false_positive_rate: float,
        false_negative_rate: float,
        delayed_detection_rate: float = 0.0,
        system_cost: float | None = None,
    ) -> float:
        fp_rate = float(false_positive_rate)
        fn_rate = float(false_negative_rate)
        delayed_rate = float(delayed_detection_rate)
        inferred_cost = self.estimate_total_cost(
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            delayed_detection_rate=delayed_rate,
        )
        total_cost = float(system_cost) if system_cost is not None else inferred_cost
        self.cost_history.append(total_cost)

        raw_signal = (
            self.fp_cost * fp_rate
            - self.fn_cost * fn_rate
            - DELAYED_DETECTION_SIGNAL_WEIGHT * self.delayed_detection_cost * delayed_rate
        )
        delta = float(self.threshold_step * raw_signal)
        tuned = tune_threshold(self.threshold, false_positive_rate=fp_rate, false_negative_rate=fn_rate, step=self.threshold_step)
        self.threshold = float(min(max(tuned + delta, MIN_THRESHOLD), MAX_THRESHOLD))
        self.threshold_history.append(self.threshold)
        return float(self.threshold)
