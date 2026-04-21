from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from attacker_engine.learning import expected_rewards, softmax_bandit, update_reward_history
from attacker_engine.strategies import STRATEGY_MARKETPLACE, normalize_strategy_mix


@dataclass
class AttackerAgent:
    strategy_mix: Dict[str, float] | None = None
    temperature: float = 0.25
    reward_history: Dict[str, List[float]] = field(default_factory=dict)
    selection_count: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strategy_mix = normalize_strategy_mix(self.strategy_mix)
        self.reward_history = {name: [float(v) for v in self.reward_history.get(name, [])] for name in STRATEGY_MARKETPLACE}
        self.selection_count = {name: int(self.selection_count.get(name, 0)) for name in STRATEGY_MARKETPLACE}

    def sample_strategy(self, rng: np.random.Generator) -> str:
        names = list(self.strategy_mix.keys())
        probs = np.array([self.strategy_mix[n] for n in names], dtype=float)
        chosen = str(rng.choice(np.array(names, dtype=object), p=probs))
        self.selection_count[chosen] = int(self.selection_count.get(chosen, 0)) + 1
        return chosen

    def update(self, rewards: Dict[str, float]) -> Dict[str, float]:
        self.reward_history = update_reward_history(self.reward_history, rewards=rewards)
        reward_means = expected_rewards(self.reward_history)
        adapted = softmax_bandit(reward_means, temperature=self.temperature)
        self.strategy_mix = normalize_strategy_mix(adapted)
        return dict(self.strategy_mix)
