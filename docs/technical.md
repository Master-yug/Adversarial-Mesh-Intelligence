# 🔬 Technical Overview — Adversarial-Mesh-Intelligence

## 1. Attacker Strategy Learning

The attacker agent is a **multi-armed bandit** — it must choose between strategies without knowing which will work best, and it learns by trying.

### Strategy Marketplace

Seven strategies are defined with fixed profiles:

```python
StrategyProfile(name, cost, stealth_level, effectiveness)
```

- **cost** — resource cost of executing the strategy
- **stealth_level** — how hard the strategy is to detect (0–1)
- **effectiveness** — raw fraud gain potential (0–1)

Example: `perfect_mimic` has `stealth=0.88`, `effectiveness=0.69`, `cost=0.34` — highly stealthy, moderately effective, expensive to run.

### Strategy Selection

Each round the attacker picks a strategy using one of two methods:

**Softmax bandit (default)**
```
P(strategy) = softmax(expected_reward / temperature)
```

Higher-reward strategies get exponentially higher probability. Temperature controls exploration: lower temperature → greedier selection.

**UCB (Upper Confidence Bound)**
```
score(strategy) = mean_reward + w × sqrt(ln(total_pulls) / pulls_for_strategy)
```

Strategies tried fewer times get a bonus, ensuring they're explored. UCB is optimal for balancing exploration vs. exploitation.

Epsilon-greedy exploration is layered on top: with probability `ε=0.08`, the attacker picks randomly regardless of learned probabilities.

### Forward Simulation (Attacker's Look-Ahead)

Before committing to a strategy, the attacker runs a **one or two-step forward simulation**:

1. It estimates the defender's likely threshold, budget, and detection intensity
2. For each candidate strategy, it computes:
   - `expected_gain` = effectiveness − threshold pressure
   - `expected_detection_probability` = weighted combination of defender signals
   - `expected_penalty` = detection probability × (cost + defender intensity)
   - `expected_utility` = expected_gain − expected_penalty

With `foresight_steps=2`, it also discounts a second round:
```
total_utility = current_utility + γ × next_utility   (γ = 0.85)
```

The attacker picks the highest-utility strategy. This is a lightweight form of **model-based reinforcement learning** — the attacker has an internal model of the defender.

### Belief Uncertainty

The attacker doesn't know the defender's state exactly. It introduces a **belief offset**:

```python
belief_offset = uncertainty_scale × (strategy_mix[s] - 1/N)
```

Strategies the attacker favors get a small optimism bias. This models **bounded rationality** — attackers are smart but not omniscient.

### Reward Update

After each round, the attacker receives a feedback signal:

```
net_reward = economic_gain − detection_penalty
```

This is used to update the reward history for each strategy (sliding window of last 48 rounds). The strategy mix is then recomputed via softmax over average rewards.

Cross-run memory persists to `memory/attacker_memory.json`, so the attacker carries knowledge across independent simulation sessions.

---

## 2. Defender Policy Optimization

The defender's goal is to **minimize total system cost** subject to a limited investigation budget.

### Cost Model

```
Total Cost = (FP_rate × 1.0)          # flagging innocent nodes
           + (FN_rate × 8.0)          # missing real attackers
           + (delayed_rate × 2.5)     # late detection penalty
           + (delayed_rate × 1.8)     # additional delay signal
           + (investigation_rate × 0.7)  # investigation overhead
```

FN cost (8.0) is 8× higher than FP cost (1.0), reflecting the reality that a missed attacker causes far more economic damage than a false alarm.

### Threshold Grid Search

Each round the defender runs an exhaustive grid search over:
- **Threshold candidates**: 15 evenly spaced values in [0.2, 0.95]
- **Budget ratios**: {3%, 5%, 7%, 10%, 14%, 18%} of total nodes
- **Selection strategies**: `risk_only`, `risk_adjusted`, `uncertainty_first`

For each combination it computes FP/FN rates and total cost. The combination with **minimum cost** is selected and applied.

### Selection Strategies

| Strategy | Score formula |
|---|---|
| `risk_only` | `fraud_score` |
| `risk_adjusted` | `fraud_score − 0.15 × uncertainty` |
| `uncertainty_first` | `fraud_score + 0.20 × uncertainty` |

`risk_adjusted` (default) penalizes uncertain predictions, reducing wasted investigations on nodes the model isn't confident about. `uncertainty_first` prioritizes exploration of ambiguous nodes.

### Adaptive Threshold Update

Between rounds, the defender also runs a gradient-style update:

```python
signal = (FP_cost × FP_rate) − (FN_cost × FN_rate) − (delay_weight × delay_rate)
delta = threshold_step × signal
new_threshold = tuned_threshold + delta
```

If FP > FN → signal is positive → threshold rises → fewer flags
If FN > FP → signal is negative → threshold falls → more flags

This keeps the defender responsive between full grid searches.

---

## 3. Equilibrium Detection

The system monitors both agents each round and detects when they've both stabilized.

### What "Equilibrium" Means

In game theory, an equilibrium is a state where no player can improve their outcome by changing strategy unilaterally. In this system:

- The attacker can't find a better strategy than its current mix
- The defender can't find a better threshold than its current setting

### Detection Method

Two signals are tracked:

**Attacker stability**: Is the strategy mix converging?
```
strategy_mix_variance = var([P(s) for all rounds in window])
stable if: max_variance < ε_strategy
```

**Defender stability**: Is the threshold stabilizing?
```
threshold_delta = |threshold[t] - threshold[t-1]|
stable if: threshold_delta < ε_threshold
```

When both conditions hold for `K` consecutive rounds (K=2 by default), the system records equilibrium.

### Why This Matters

Equilibrium detection provides a **convergence guarantee**: once reached, the system's fraud leakage, cost, and detection rate are as good as they'll get under current conditions. This is the system's security baseline.

If conditions change (new attack campaign, new network topology), the system exits equilibrium and re-converges.

---

## 4. Uncertainty Modeling

The fraud model doesn't just output a score — it outputs **uncertainty** alongside it.

### Sources of Uncertainty

1. **Ensemble disagreement** — RandomForest uses many trees; the variance across tree predictions measures model confidence
2. **Feature noise** — measurement error, packet loss, and partial visibility introduce input uncertainty
3. **Label noise** — some training samples have noisy or delayed labels; the model learns under this ambiguity
4. **Correlated noise** — shared noise across features (latent noise) makes patterns harder to separate

### How Uncertainty Is Used

- **In the API**: returned as `uncertainty_score` (0–1) alongside `fraud_score`
- **In the defender**: used by `uncertainty_first` and `risk_adjusted` selection strategies
- **In training**: ambiguous samples (flagged `is_ambiguous`) are handled with semi-supervised logic; delayed labels contribute to a `delayed_detection_rate` signal

### Partial Observability

The API supports **partial observability mode** — simulating production conditions where not all signals are available:

- Random feature dropout (some sensors fail)
- Delayed evidence (GPS signals arrive late)
- Conflicting evidence (two sensors disagree)

This makes the system robust to the messy reality of live network monitoring.

---

## 5. Simulation Realism

The network simulation is designed to be hard to game:

- **Honest anomaly injection** — honest nodes sometimes behave strangely (mimicking attack patterns), creating hard negative examples
- **Semi-supervised labeling** — some labels arrive late, some never arrive (simulating real-world data pipelines where fraud is confirmed only after investigation)
- **Temporal drift** — node behavior drifts over time, so models trained on early data degrade on later data
- **Distribution shift testing** — the system explicitly trains on one difficulty level and evaluates on a harder one, testing generalization

### Difficulty Levels

| Level | What's harder |
|---|---|
| `easy` | Clean signals, low noise, stable behavior |
| `medium` | Moderate noise, some temporal drift |
| `hard` | High noise, correlated features, label delays |
| `extreme` | Maximum noise, strong distribution shift, ambiguous labels |

---

## 6. Feature Engineering

18 base features capture behavioral signals at each timestep:

- Latency statistics (forward, reverse, ratio)
- GPS consistency (claimed vs. inferred location drift)
- Peer graph metrics (clustering coefficient, reciprocity)
- Trust score trajectory
- Response time consistency
- Packet loss patterns
- Historical fraud score trend

Each feature is aggregated into `mean`, `std`, and `last` across the observation window — giving **54 final model inputs** per node.

The aggregation is intentionally non-naive: recent-weighted partial history and rolling instability summaries are mapped into these suffixes to preserve temporal instability while keeping a fixed model schema.
