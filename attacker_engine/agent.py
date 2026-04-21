from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np

from attacker_engine.learning import (
    apply_epsilon_exploration,
    expected_rewards,
    learning_step,
    softmax_bandit,
    strategy_reward,
    update_reward_history,
)
from attacker_engine.strategies import STRATEGY_MARKETPLACE, normalize_strategy_mix


@dataclass
class AttackerAgent:
    strategy_mix: Dict[str, float] | None = None
    temperature: float = 0.25
    epsilon: float = 0.08
    use_ucb_selection: bool = False
    ucb_exploration_weight: float = 0.9
    reward_history: Dict[str, List[float]] = field(default_factory=dict)
    selection_count: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strategy_mix = normalize_strategy_mix(self.strategy_mix)
        self.reward_history = {name: [float(v) for v in self.reward_history.get(name, [])] for name in STRATEGY_MARKETPLACE}
        self.selection_count = {name: int(self.selection_count.get(name, 0)) for name in STRATEGY_MARKETPLACE}

    def sample_strategy(self, rng: np.random.Generator) -> str:
        names = list(self.strategy_mix.keys())
        if self.use_ucb_selection:
            chosen = self._sample_strategy_with_ucb(names=names, rng=rng)
        elif rng.random() < float(np.clip(self.epsilon, 0.0, 1.0)):
            chosen = str(rng.choice(np.array(names, dtype=object)))
        else:
            probs = np.array([self.strategy_mix[n] for n in names], dtype=float)
            chosen = str(rng.choice(np.array(names, dtype=object), p=probs))
        self.selection_count[chosen] = int(self.selection_count.get(chosen, 0)) + 1
        return chosen

    def _sample_strategy_with_ucb(self, names: List[str], rng: np.random.Generator) -> str:
        total_count = float(sum(self.selection_count.get(name, 0) for name in names))
        if total_count <= 0:
            return str(rng.choice(np.array(names, dtype=object)))
        reward_means = expected_rewards(self.reward_history)
        scores: Dict[str, float] = {}
        unexplored: List[str] = []
        for name in names:
            pulls = float(max(self.selection_count.get(name, 0), 0))
            if pulls <= 0:
                unexplored.append(name)
                continue
            confidence = float(np.sqrt(np.log(total_count + 1.0) / pulls))
            scores[name] = float(reward_means.get(name, 0.0) + self.ucb_exploration_weight * confidence)
        if unexplored:
            return str(rng.choice(np.array(unexplored, dtype=object)))
        best = max(scores.values())
        best_names = [name for name, score in scores.items() if score == best]
        return str(rng.choice(np.array(best_names, dtype=object)))

    def update(
        self,
        rewards: Mapping[str, float] | None = None,
        strategy_feedback: Mapping[str, Mapping[str, float]] | None = None,
    ) -> Dict[str, float]:
        feedback = strategy_feedback
        if feedback is None:
            feedback = {
                str(name): {
                    "economic_gain": float(value),
                    "detection_penalty": 0.0,
                }
                for name, value in (rewards or {}).items()
            }

        self.reward_history, self.selection_count, adapted = learning_step(
            reward_history=self.reward_history,
            usage_count=self.selection_count,
            strategy_feedback=feedback,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )
        self.strategy_mix = normalize_strategy_mix(adapted)
        return dict(self.strategy_mix)

    def strategy_net_rewards(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in STRATEGY_MARKETPLACE:
            history = self.reward_history.get(name, [])
            out[name] = float(np.mean(history)) if history else 0.0
        return out

    def observe_single_strategy_feedback(self, strategy_name: str, economic_gain: float, detection_penalty: float) -> None:
        net = strategy_reward(economic_gain=economic_gain, detection_penalty=detection_penalty)
        self.reward_history = update_reward_history(self.reward_history, rewards={strategy_name: net})
        reward_means = expected_rewards(self.reward_history)
        adapted = softmax_bandit(reward_means, temperature=self.temperature)
        explored = apply_epsilon_exploration(adapted, epsilon=self.epsilon)
        self.strategy_mix = normalize_strategy_mix(explored)
