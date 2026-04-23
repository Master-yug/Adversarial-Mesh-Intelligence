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

# Defender-response weights used by the attacker for one-step utility simulation.
# Larger weights increase expected detection probability under that signal.
D_THRESHOLD_WEIGHT = 0.55
D_BUDGET_WEIGHT = 0.35
D_INTENSITY_WEIGHT = 0.30
D_STEALTH_WEIGHT = 0.20
D_CONGESTION_WEIGHT = 0.15
D_TRUST_WEIGHT = 0.10
DISCOUNT_FACTOR = 0.85

@dataclass(frozen=True)
class StrategicSimulation:
    """Forward simulation summary for one attacker strategy against predicted defender state."""

    strategy: str
    expected_gain: float
    expected_detection_probability: float
    expected_penalty: float
    expected_utility: float
    expected_defender_threshold: float


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

    def simulate_outcome(
        self,
        strategy: str,
        defender_state: Mapping[str, float],
        environment_state: Mapping[str, float],
        uncertainty_scale: float = 0.08,
        foresight_steps: int = 1,
    ) -> StrategicSimulation:
        profile = STRATEGY_MARKETPLACE.get(strategy)
        if profile is None:
            return StrategicSimulation(
                strategy=strategy,
                expected_gain=0.0,
                expected_detection_probability=1.0,
                expected_penalty=1.0,
                expected_utility=-1.0,
                expected_defender_threshold=float(defender_state.get("threshold", 0.5)),
            )

        threshold = float(np.clip(defender_state.get("threshold", 0.5), 0.2, 0.95))
        budget_ratio = float(np.clip(defender_state.get("budget_ratio", 0.07), 0.0, 0.6))
        defender_intensity = float(np.clip(defender_state.get("defender_intensity", 0.5), 0.0, 1.0))
        leakage_pressure = float(np.clip(environment_state.get("fraud_leakage", 0.20), 0.0, 1.0))
        fraud_profit_pressure = float(max(environment_state.get("fraud_profit", 0.0), 0.0))
        congestion = float(np.clip(environment_state.get("network_congestion", 0.15), 0.0, 1.0))
        trust_shift = float(np.clip(environment_state.get("trust_shift", 0.0), -1.0, 1.0))
        defender_response_shift = float(np.clip(environment_state.get("defender_response_shift", 0.0), -0.2, 0.2))

        belief_offset = float(
            uncertainty_scale
            * (
                float(self.strategy_mix.get(strategy, 0.0))
                - (1.0 / max(len(self.strategy_mix), 1))
            )
        )
        perceived_threshold = float(np.clip(threshold + 0.08 * belief_offset, 0.2, 0.95))
        perceived_budget = float(np.clip(budget_ratio + 0.10 * belief_offset, 0.0, 0.6))
        perceived_intensity = float(np.clip(defender_intensity + 0.10 * belief_offset, 0.0, 1.0))

        current_threshold = float(
            np.clip(
                perceived_threshold + defender_response_shift + 0.08 * perceived_intensity - 0.04 * leakage_pressure,
                0.2,
                0.95,
            )
        )
        current_detection_probability = float(
            np.clip(
                D_THRESHOLD_WEIGHT * current_threshold
                + D_BUDGET_WEIGHT * perceived_budget
                + D_INTENSITY_WEIGHT * perceived_intensity
                + D_STEALTH_WEIGHT * (1.0 - profile.stealth_level)
                + D_CONGESTION_WEIGHT * congestion
                - D_TRUST_WEIGHT * trust_shift
                + belief_offset,
                0.01,
                0.99,
            )
        )
        current_expected_gain = float(
            0.75 * profile.effectiveness
            + 0.55 * leakage_pressure
            + 0.05 * fraud_profit_pressure
            - 0.25 * current_threshold
        )
        current_expected_penalty = float(
            current_detection_probability * (0.9 + profile.cost + 0.25 * perceived_intensity + 0.20 * congestion)
        )
        current_expected_utility = float(current_expected_gain - current_expected_penalty)

        if foresight_steps <= 1:
            next_expected_gain = current_expected_gain
            next_detection_probability = current_detection_probability
            next_expected_penalty = current_expected_penalty
            expected_threshold = current_threshold
            expected_utility = current_expected_utility
        else:
            anticipated_threshold = float(
                np.clip(
                    current_threshold + 0.06 * current_detection_probability - 0.03 * leakage_pressure,
                    0.2,
                    0.95,
                )
            )
            anticipated_budget = float(np.clip(perceived_budget + 0.08 * current_detection_probability, 0.0, 0.6))
            anticipated_intensity = float(np.clip(perceived_intensity + 0.10 * current_detection_probability, 0.0, 1.0))
            next_leakage_pressure = float(np.clip(leakage_pressure + 0.10 * max(current_expected_utility, 0.0), 0.0, 1.0))
            next_detection_probability = float(
                np.clip(
                    D_THRESHOLD_WEIGHT * anticipated_threshold
                    + D_BUDGET_WEIGHT * anticipated_budget
                    + D_INTENSITY_WEIGHT * anticipated_intensity
                    + D_STEALTH_WEIGHT * (1.0 - profile.stealth_level)
                    + D_CONGESTION_WEIGHT * congestion
                    - D_TRUST_WEIGHT * trust_shift
                    + 0.5 * belief_offset,
                    0.01,
                    0.99,
                )
            )
            next_expected_gain = float(
                0.75 * profile.effectiveness
                + 0.55 * next_leakage_pressure
                + 0.05 * fraud_profit_pressure
                - 0.25 * anticipated_threshold
            )
            next_expected_penalty = float(
                next_detection_probability * (0.9 + profile.cost + 0.25 * anticipated_intensity + 0.20 * congestion)
            )
            next_expected_utility = float(next_expected_gain - next_expected_penalty)
            expected_threshold = anticipated_threshold
            expected_utility = float(current_expected_utility + DISCOUNT_FACTOR * next_expected_utility)

        expected_gain = float(current_expected_gain + DISCOUNT_FACTOR * next_expected_gain)
        expected_detection_probability = float(
            np.clip(0.5 * current_detection_probability + 0.5 * next_detection_probability, 0.01, 0.99)
        )
        expected_penalty = float(current_expected_penalty + DISCOUNT_FACTOR * next_expected_penalty)
        return StrategicSimulation(
            strategy=strategy,
            expected_gain=expected_gain,
            expected_detection_probability=expected_detection_probability,
            expected_penalty=expected_penalty,
            expected_utility=expected_utility,
            expected_defender_threshold=expected_threshold,
        )

    def choose_strategic_strategy(
        self,
        defender_state: Mapping[str, float],
        environment_state: Mapping[str, float],
        rng: np.random.Generator,
        foresight_steps: int = 2,
        uncertainty_scale: float = 0.08,
    ) -> tuple[str, Dict[str, StrategicSimulation]]:
        names = list(self.strategy_mix.keys())
        if not names:
            names = list(STRATEGY_MARKETPLACE.keys())
        simulations = {
            name: self.simulate_outcome(
                strategy=name,
                defender_state=defender_state,
                environment_state=environment_state,
                uncertainty_scale=uncertainty_scale,
                foresight_steps=foresight_steps,
            )
            for name in names
        }
        best_utility = max(sim.expected_utility for sim in simulations.values())
        best_names = [name for name, sim in simulations.items() if sim.expected_utility == best_utility]
        if rng.random() < float(np.clip(self.epsilon, 0.0, 1.0)):
            chosen = str(rng.choice(np.array(names, dtype=object)))
        else:
            chosen = str(rng.choice(np.array(best_names, dtype=object)))
        self.selection_count[chosen] = int(self.selection_count.get(chosen, 0)) + 1
        return chosen, simulations

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
