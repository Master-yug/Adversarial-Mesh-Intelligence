from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class StrategyProfile:
    """Marketplace definition for attacker strategies."""

    name: str
    cost: float
    stealth_level: float
    effectiveness: float


@dataclass
class StrategyLearningState:
    """Tracks per-strategy rewards and adaptive attacker strategy allocation."""

    strategy_mix: Dict[str, float]
    reward_history: Dict[str, List[float]] = field(default_factory=dict)
    selection_count: Dict[str, int] = field(default_factory=dict)
    temperature: float = 0.25

    def __post_init__(self) -> None:
        self.strategy_mix = normalize_strategy_mix(self.strategy_mix)
        self.reward_history = {
            name: [float(v) for v in self.reward_history.get(name, [])]
            for name in STRATEGY_MARKETPLACE
        }
        self.selection_count = {name: int(self.selection_count.get(name, 0)) for name in STRATEGY_MARKETPLACE}


STRATEGY_MARKETPLACE: Dict[str, StrategyProfile] = {
    "low_and_slow": StrategyProfile(name="low_and_slow", cost=0.22, stealth_level=0.84, effectiveness=0.46),
    "burst_attack": StrategyProfile(name="burst_attack", cost=0.30, stealth_level=0.38, effectiveness=0.78),
    "camouflage": StrategyProfile(name="camouflage", cost=0.28, stealth_level=0.79, effectiveness=0.58),
    "perfect_mimic": StrategyProfile(name="perfect_mimic", cost=0.34, stealth_level=0.88, effectiveness=0.69),
    "slow_drift": StrategyProfile(name="slow_drift", cost=0.27, stealth_level=0.81, effectiveness=0.61),
    "decoy_attacker": StrategyProfile(name="decoy_attacker", cost=0.24, stealth_level=0.73, effectiveness=0.52),
    "mixed_cluster": StrategyProfile(name="mixed_cluster", cost=0.31, stealth_level=0.67, effectiveness=0.73),
}


def normalize_strategy_mix(mix: Dict[str, float] | None) -> Dict[str, float]:
    """Normalize and filter a strategy mix to known marketplace strategies."""
    if not mix:
        return {name: 1.0 / len(STRATEGY_MARKETPLACE) for name in STRATEGY_MARKETPLACE}
    filtered = {k: float(v) for k, v in mix.items() if k in STRATEGY_MARKETPLACE and float(v) >= 0.0}
    if not filtered:
        return {name: 1.0 / len(STRATEGY_MARKETPLACE) for name in STRATEGY_MARKETPLACE}
    total = float(sum(filtered.values()))
    if total <= 0:
        return {name: 1.0 / len(STRATEGY_MARKETPLACE) for name in STRATEGY_MARKETPLACE}
    return {k: float(v / total) for k, v in filtered.items()}


def evolve_strategy_mix(
    current_mix: Dict[str, float],
    mutation_rate: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Mutate and renormalize a strategy mix to emulate emerging attacker behavior."""
    out = dict(normalize_strategy_mix(current_mix))
    for key in list(out.keys()):
        if rng.random() < mutation_rate:
            delta = float(rng.normal(0.0, 0.08))
            out[key] = max(0.0, out[key] + delta)
    return normalize_strategy_mix(out)


def softmax_strategy_mix(
    rewards: Dict[str, float],
    temperature: float = 0.25,
) -> Dict[str, float]:
    temp = max(float(temperature), 1e-3)
    ordered = sorted(STRATEGY_MARKETPLACE.keys())
    scores = np.array([float(rewards.get(name, 0.0)) for name in ordered], dtype=float)
    shifted = scores - np.max(scores)
    probs = np.exp(shifted / temp)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return {name: 1.0 / len(ordered) for name in ordered}
    return {name: float(prob / total) for name, prob in zip(ordered, probs)}


def update_learning_state(
    state: StrategyLearningState,
    strategy_rewards: Dict[str, float],
    history_limit: int = 48,
) -> StrategyLearningState:
    history_cap = max(int(history_limit), 1)
    for strategy_name in STRATEGY_MARKETPLACE:
        reward = float(strategy_rewards.get(strategy_name, 0.0))
        history = list(state.reward_history.get(strategy_name, []))
        history.append(reward)
        state.reward_history[strategy_name] = history[-history_cap:]
        state.selection_count[strategy_name] = int(state.selection_count.get(strategy_name, 0)) + 1

    expected_rewards = {
        name: float(np.mean(state.reward_history.get(name, [0.0])))
        for name in STRATEGY_MARKETPLACE
    }
    state.strategy_mix = normalize_strategy_mix(
        softmax_strategy_mix(expected_rewards, temperature=state.temperature)
    )
    return state
