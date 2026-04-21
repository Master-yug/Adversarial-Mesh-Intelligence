from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from defender_engine.policies import tune_threshold


@dataclass
class DefenderAgent:
    threshold: float = 0.5
    budget: int = 10

    def select_nodes(self, fraud_scores: Dict[int, float], node_selection: Iterable[int] | None = None) -> List[int]:
        candidates = list(node_selection) if node_selection is not None else list(fraud_scores.keys())
        ranked = sorted(candidates, key=lambda n: float(fraud_scores.get(int(n), 0.0)), reverse=True)
        selected = [int(node_id) for node_id in ranked if float(fraud_scores.get(int(node_id), 0.0)) >= float(self.threshold)]
        return selected[: max(int(self.budget), 0)]

    def update_policy(self, false_positive_rate: float, false_negative_rate: float) -> float:
        self.threshold = tune_threshold(self.threshold, false_positive_rate=false_positive_rate, false_negative_rate=false_negative_rate)
        return float(self.threshold)
