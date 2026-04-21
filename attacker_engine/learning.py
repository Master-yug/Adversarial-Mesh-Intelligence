from __future__ import annotations

from typing import Dict, List

import numpy as np


def update_reward_history(reward_history: Dict[str, List[float]], rewards: Dict[str, float], history_limit: int = 64) -> Dict[str, List[float]]:
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
