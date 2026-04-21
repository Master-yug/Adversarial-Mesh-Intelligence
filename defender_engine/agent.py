from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping

from defender_engine.policies import tune_threshold

UNCERTAINTY_PENALTY_WEIGHT = 0.15
DELAYED_DETECTION_SIGNAL_WEIGHT = 0.5
MIN_THRESHOLD = 0.2
MAX_THRESHOLD = 0.95

@dataclass
class DefenderAgent:
    threshold: float = 0.5
    budget: int | None = None
    budget_ratio: float = 0.07
    min_budget: int = 1
    fp_cost: float = 1.0
    fn_cost: float = 8.0
    delayed_detection_cost: float = 2.5
    threshold_step: float = 0.03
    threshold_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)

    def _effective_budget(self, candidate_count: int) -> int:
        if self.budget is not None:
            return max(int(self.budget), 0)
        ratio_budget = int(round(float(self.budget_ratio) * int(candidate_count)))
        return max(int(self.min_budget), ratio_budget)

    def select_nodes(
        self,
        fraud_scores: Mapping[int, float],
        node_selection: Iterable[int] | None = None,
        uncertainty_scores: Mapping[int, float] | None = None,
    ) -> List[int]:
        candidates = list(node_selection) if node_selection is not None else list(fraud_scores.keys())
        uncertainty_scores = uncertainty_scores or {}
        ranked = sorted(
            candidates,
            key=lambda n: (
                float(fraud_scores.get(int(n), 0.0))
                - UNCERTAINTY_PENALTY_WEIGHT * float(uncertainty_scores.get(int(n), 0.0))
            ),
            reverse=True,
        )
        selected = [int(node_id) for node_id in ranked if float(fraud_scores.get(int(node_id), 0.0)) >= float(self.threshold)]
        return selected[: self._effective_budget(candidate_count=len(candidates))]

    def estimate_total_cost(
        self,
        false_positive_rate: float,
        false_negative_rate: float,
        delayed_detection_rate: float = 0.0,
    ) -> float:
        fp_component = self.fp_cost * float(false_positive_rate)
        fn_component = self.fn_cost * float(false_negative_rate)
        delayed_component = self.delayed_detection_cost * float(delayed_detection_rate)
        return float(fp_component + fn_component + delayed_component)

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
