#  System Architecture — Adversarial-Mesh-Intelligence

## Overview

Adversarial-Mesh-Intelligence is built as a **closed-loop adversarial system** — a continuous cycle where an attacker agent and a defender agent compete, learn, and push each other toward a strategic equilibrium.

The architecture has five layers:

```
┌──────────────────────────────────────────────────────────────┐
│                    SIMULATION LAYER                          │
│   Generates the decentralized network (nodes, GPS, latency)  │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   FEATURE LAYER                              │
│   Extracts 54 behavioral signals per node per timestep       │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                DETECTION (ML) LAYER                          │
│   Fraud model scores every node 0→1                          │
└──────────┬─────────────────────────┬─────────────────────────┘
           │                         │
┌──────────▼──────────┐   ┌──────────▼──────────────────────────┐
│   ATTACKER ENGINE   │   │        DEFENDER ENGINE              │
│  Learns strategies  │   │  Optimizes threshold + budget       │
│  Simulates defender │   │  Reacts to false positives          │
└──────────┬──────────┘   └──────────┬──────────────────────────┘
           │                         │
┌──────────▼─────────────────────────▼─────────────────────────┐
│                  CORE ENGINE (LOOP)                          │
│  Orchestrates each round, detects equilibrium                │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                  EVALUATION LAYER                            │
│  Metrics, economic impact, plots, API                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### `simulation/` — The Decentralized Network

Creates a synthetic but realistic decentralized peer network. Every run generates:

- **Honest nodes** — behave normally with small amounts of natural measurement noise
- **Naive attacker nodes** — use simple fraud without adaptation
- **Smart attacker nodes** — adapt based on detection feedback

Key controls include node count, honest/attacker ratio, GPS noise, packet loss, partial visibility, and difficulty level (`easy` → `extreme`).

The environment also supports **scenario injections**: sudden attack surges, coordinated cluster campaigns, regional takeovers, and stealth infiltration campaigns.

---

### `features/` — Behavioral Signals

Extracts **18 base behavioral features** per node per timestep, then aggregates them into:
- `<feature>_mean` — average behavior
- `<feature>_std` — behavioral volatility
- `<feature>_last` — most recent reading

This gives **54 features** per node — the inputs to the fraud detection model.

Features capture things like latency patterns, GPS consistency, peer cluster behavior, trust stability, and temporal drift.

---

### `modeling/` — Fraud Detection Model

Trains and benchmarks multiple ML models on the labeled simulation data:

- `RandomForestClassifier`
- `XGBClassifier`
- Heuristic baseline
- Rolling-window RF (sequence-aware)

The best model and optimal threshold are saved as `model.pkl` and used live in subsequent simulation rounds.

---

### `attacker_engine/` — The Attacker Agent

The attacker is a **learning agent** that picks from a marketplace of 7 fraud strategies:

```
Strategy        Stealth   Effectiveness   Cost
─────────────── ───────   ─────────────   ────
low_and_slow     0.84         0.46        0.22   ← quiet but weak
burst_attack     0.38         0.78        0.30   ← loud but powerful
camouflage       0.79         0.58        0.28
perfect_mimic    0.88         0.69        0.34   ← hardest to catch
slow_drift       0.81         0.61        0.27
decoy_attacker   0.73         0.52        0.24
mixed_cluster    0.67         0.73        0.31
```

**How the attacker decides:**

1. For each candidate strategy, it runs a **forward simulation** — estimating how the defender would likely respond (given current threshold, budget, and intensity)
2. It computes an **expected utility** = expected gain − expected detection penalty
3. It picks the highest-utility strategy (with epsilon-exploration for discovery)
4. After the round, it updates its strategy probability mix using a **softmax bandit** — strategies that earned more reward get higher probability next time

The attacker also uses **Upper Confidence Bound (UCB)** selection optionally, balancing exploitation vs. exploration.

---

### `defender_engine/` — The Defender Agent

The defender's job: flag fraud nodes without burning honest users.

It has three levers:

| Lever | What it controls |
|---|---|
| `threshold` | Minimum fraud score to flag a node (0.2–0.95) |
| `budget_ratio` | What fraction of nodes can be investigated |
| `selection_strategy` | How to rank candidates (`risk_only`, `risk_adjusted`, `uncertainty_first`) |

**How the defender optimizes:**

Each round it runs a **grid search** across threshold × budget ratio × selection strategy combinations, picking the combination that minimizes total system cost:

```
Total Cost = (FP rate × FP cost)
           + (FN rate × FN cost)
           + (delayed detection rate × delay cost)
           + (investigation rate × investigation cost)
```

FN cost (8.0) is intentionally much higher than FP cost (1.0) — missing a real attacker is far more expensive than flagging an innocent node.

The defender also tracks `threshold_history` and `cost_history` to inform equilibrium detection.

---

### `core_engine/loop.py` — The Adversarial Loop

The main engine that runs one closed-loop iteration:

```
┌─ AdversarialFraudEnvironment ─────────────────────────┐
│                                                        │
│  1. Build simulation at current difficulty             │
│  2. Extract features                                   │
│  3. Train or update fraud detection model              │
│  4. Score all nodes → fraud_scores                     │
│  5. Defender selects nodes to investigate              │
│  6. Compute FP/FN/cost                                 │
│  7. Attacker observes leakage, updates strategy mix    │
│  8. Defender observes cost, updates threshold          │
│  9. Check for equilibrium                              │
│ 10. Log round results                                  │
└────────────────────────────────────────────────────────┘
```

**Equilibrium detection** works by measuring whether:
- The attacker's strategy mix has stopped shifting (variance below threshold)
- The defender's threshold has stopped shifting (delta below threshold)

When both conditions hold for consecutive rounds, the system is in equilibrium.

**Hard-negative mining**: Failed detections from each round (missed attackers, slow detections) are upweighted in the next round's training set — the model continuously focuses on its own blind spots.

---

### `core_engine/orchestrator.py` — Full Pipeline

End-to-end run:
1. Baseline simulation (no adaptation)
2. Feature extraction
3. Model training (RF + XGBoost benchmark)
4. Adaptive simulation (model-in-the-loop)
5. Detection metrics
6. Economic impact analysis
7. Visualization of all results

Produces `model.pkl` and diagnostic plots.

---

### `evaluation/` — Measuring What Matters

**`metrics.py`** — Detection quality:
- Per-node: first detection timestep, detection consistency, detection delay
- Summary: average detection delay, % caught within N steps, false positive rate over time

**`economic.py`** — Real-world cost:
- Total fraud profit earned by undetected attackers
- Fraud reduction after detection kicks in
- % of total rewards lost to fraud
- Regional fraud concentration
- High-value attack success rate

**`visualization.py`** — What judges see:
- Strategy distribution stacked area chart
- Defender threshold trajectory
- System cost over time
- Fraud leakage over time
- Equilibrium detection signal

---

### `api/` — Real-Time Scoring

FastAPI endpoint for scoring live node observations:

```
POST /score-node
Input:  latencies, peers, claimed_location, history, ...
Output: fraud_score (0-1), risk_label, trend, confidence, reasons
```

Supports partial observability (dropout, delayed evidence) for realistic production conditions.

---

## Data Flow Summary

```
Network Simulation
      │
      ▼
Feature Extraction (54 signals per node)
      │
      ▼
ML Fraud Scoring (RandomForest / XGBoost)
      │
      ├──────────────────────────────┐
      ▼                              ▼
Attacker Agent                 Defender Agent
(picks strategy based       (investigates nodes based
 on expected utility)        on threshold + budget)
      │                              │
      └──────────┬───────────────────┘
                 ▼
         Reward / Cost Signals
                 │
         ┌───────▼────────┐
         │  Both agents   │
         │  update their  │
         │  policies      │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Equilibrium   │
         │  check         │
         └────────────────┘
```
