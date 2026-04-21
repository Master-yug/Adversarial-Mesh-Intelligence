from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np


def strategy_reward(economic_gain: float, detection_penalty: float) -> float:
    return float(economic_gain) - float(detection_penalty)


def update_reward_history(
    reward_history: Dict[str, List[float]],
    rewards: Mapping[str, float],
    history_limit: int = 64,
) -> Dict[str, List[float]]:
    out = {k: [float(v) for v in vals] for k, vals in reward_history.items()}
    cap = max(int(history_limit), 1)
    for strategy, reward in rewards.items():
        out.setdefault(strategy, []).append(float(reward))
        out[strategy] = out[strategy][-cap:]
    return out


def expected_rewards(reward_history: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in reward_history.items()}


def softmax_bandit(rewards: Dict[str, float], temperature: float = 0.25) -> Dict[str, float]:
    if not rewards:
        return {}
    names = sorted(rewards.keys())
    temp = max(float(temperature), 1e-3)
    values = np.array([float(rewards[n]) for n in names], dtype=float)
    values = values - np.max(values)
    probs = np.exp(values / temp)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0:
        uniform = 1.0 / max(len(names), 1)
        return {n: uniform for n in names}
    return {n: float(p / total) for n, p in zip(names, probs)}


def apply_epsilon_exploration(strategy_mix: Dict[str, float], epsilon: float) -> Dict[str, float]:
    if not strategy_mix:
        raise ValueError("strategy_mix cannot be empty: received empty dictionary in apply_epsilon_exploration")
    eps = float(np.clip(epsilon, 0.0, 1.0))
    names = sorted(strategy_mix.keys())
    uniform = 1.0 / max(len(names), 1)
    explored = {name: (1.0 - eps) * float(strategy_mix.get(name, 0.0)) + eps * uniform for name in names}
    total = float(sum(explored.values()))
    if total <= 0.0 or not np.isfinite(total):
        return {name: uniform for name in names}
    return {name: float(value / total) for name, value in explored.items()}


def update_usage_count(
    usage_count: Dict[str, int],
    strategy_feedback: Mapping[str, Mapping[str, float]],
) -> Dict[str, int]:
    out = {k: int(v) for k, v in usage_count.items()}
    for strategy_name in strategy_feedback.keys():
        out[strategy_name] = int(out.get(strategy_name, 0)) + 1
    return out


def learning_step(
    reward_history: Dict[str, List[float]],
    usage_count: Dict[str, int],
    strategy_feedback: Mapping[str, Mapping[str, float]],
    temperature: float,
    epsilon: float = 0.0,
    history_limit: int = 64,
) -> tuple[Dict[str, List[float]], Dict[str, int], Dict[str, float]]:
    rewards = {
        name: strategy_reward(
            economic_gain=float(feedback.get("economic_gain", 0.0)),
            detection_penalty=float(feedback.get("detection_penalty", 0.0)),
        )
        for name, feedback in strategy_feedback.items()
    }
    next_history = update_reward_history(reward_history, rewards=rewards, history_limit=history_limit)
    next_usage = update_usage_count(usage_count, strategy_feedback=strategy_feedback)
    expected = expected_rewards(next_history)
    policy = softmax_bandit(expected, temperature=temperature)
    explored = apply_epsilon_exploration(policy, epsilon=epsilon)
    return next_history, next_usage, explored
