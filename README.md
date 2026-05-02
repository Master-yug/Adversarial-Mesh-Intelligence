#  Adversarial-Mesh-Intelligence

 **The first game-theoretic fraud detection engine built for decentralized networks — where attackers learn, defenders adapt, and the system finds its own equilibrium.**

---

##  The Problem

Decentralized networks — GPS-reward systems, DePIN infrastructure, Web3 peer networks — are under attack.

Fraudsters **spoof their location**, fake their identity, or flood the network with fake nodes to steal rewards they never earned. This is called **GPS spoofing**, **Sybil attacks**, or **location fraud**.

**Why current systems fail:**
- They use static rules that attackers learn to bypass in days
- They flag too many honest users (false positives kill trust)
- They react slowly — fraud happens in real time
- They don't model attacker *strategy* — just attacker *behavior*

The result? Millions in fraud losses, degraded network quality, and honest participants getting squeezed out.

---

##  The Solution

**Adversarial-Mesh-Intelligence** is a closed-loop adversarial simulation engine that treats fraud detection as a **strategic game** — not a static filter.

Two intelligent agents compete in real time:

-  **The Attacker Agent** — learns which fraud strategies succeed, adapts its mix every round, and simulates what the defender will do *before* it acts
-  **The Defender Agent** — optimizes its detection threshold and investigation budget, adjusting after every round based on cost signals

The system runs, learns, and evolves until it reaches **equilibrium** — the point where neither side can gain by changing strategy. That stable state tells you *exactly* how resilient your network is.

**Why this is different:**
 Most fraud detectors are trained once and deployed. Ours keeps learning. It simulates how attackers evolve, and builds a defender that evolves with them.

---

##  How It Works

```
1. Simulate Network
   └─ Generate hundreds of nodes (honest + attacker) with realistic behavior,
      GPS signals, latency, trust scores, and noise.

2. Attackers Strike
   └─ The attacker agent chooses from 7 strategies (burst attack, camouflage,
      perfect mimic, low-and-slow, etc.) using learned probabilities.
      It simulates defender reactions before committing.

3. Defender Detects
   └─ An ML model scores every node (0–1 fraud probability).
      The defender selects which nodes to investigate using a cost-optimized
      threshold and a bounded investigation budget.

4. Both Agents Learn
   └─ The attacker updates its strategy mix based on what worked.
      The defender adjusts its threshold and budget based on false positives,
      missed attackers, and total system cost.

5. System Evolves
   └─ Each round feeds into the next. The environment updates.
      Attack scenarios shift (surges, stealth campaigns, coordinated clusters).

6. Equilibrium Is Detected
   └─ When the attacker's strategy mix and the defender's threshold stop
      changing significantly, the system has stabilized. This is your
      security baseline.
```

---

##  Key Innovations

| Innovation | What It Means |
|---|---|
| **Multi-Agent Architecture** | Attacker and defender are independent intelligent agents, not rule engines |
| **Game-Theoretic Modeling** | Attacker simulates defender reactions before choosing a strategy |
| **7-Strategy Marketplace** | Attackers pick from low-and-slow, burst, camouflage, perfect mimic, and more |
| **Softmax Bandit Learning** | Strategy probabilities update via reinforcement — attackers remember what worked |
| **Equilibrium Detection** | System identifies when both agents have stabilized — your real security floor |
| **Cost-Aware Defense** | Defender minimizes a full cost function: false positives + missed fraud + investigation overhead |
| **Curriculum Simulation** | Difficulty scales across rounds; hard examples are mined and reused |
| **Cross-Run Memory** | Strategy history persists between runs, enabling long-term learning |

---

##  Results & Impact

Running the system across adversarial scenarios reveals consistent behavior:

- **Fraud leakage** (undetected attacks reaching the network) drops significantly as the defender adapts
- **Detection speed** improves over rounds — attackers that bypassed early models get caught faster in later rounds
- **System cost** (combined false positives + missed fraud) reaches a minimum near equilibrium
- **Strategy diversity** in the attacker mix narrows as unprofitable strategies are abandoned

In benchmark runs:
- The adaptive closed-loop model **outperforms static RandomForest and XGBoost baselines** when evaluated under distribution shift (trained on medium difficulty, tested on extreme difficulty)
- The system reaches **equilibrium within 4–8 rounds** under most scenarios

---

##  Demo Instructions

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the live demo

```bash
# Balanced environment (default)
python run_demo.py --scenario normal

# Sudden burst attack surge
python run_demo.py --scenario spike

# Long-term stealth infiltration
python run_demo.py --scenario stealth

# Organized cluster-based assault
python run_demo.py --scenario coordinated
```

### What you'll see

- A round-by-round narrative printed to your terminal with color-coded metrics
- Attacker strategy mix evolution (which fraud tactics dominate)
- Defender threshold adjustments over time
- Fraud leakage and system cost per round
- Equilibrium detection signal
- Plots saved to `demo_output/` including:
  - `strategy_distribution.png` — attacker strategy evolution
  - `defender_threshold.png` — threshold adaptation
  - `fraud_leakage.png` — undetected fraud over time
  - `system_cost.png` — total defense cost trajectory
  - `equilibrium.png` — stability signal

### Full pipeline (train + evaluate)

```bash
python -m core_engine.orchestrator
```

### API scoring server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# POST /score-node with node observations → returns fraud_score, risk_label, trend, reasons
```

---

##  Project Structure

```
adversarial-mesh-intelligence/
│
├── attacker_engine/        # Attacker AI — strategy learning, forward simulation, bandit updates
│   ├── agent.py            # AttackerAgent: UCB/softmax strategy selection, utility simulation
│   ├── strategies.py       # 7-strategy marketplace with stealth/effectiveness/cost profiles
│   └── learning.py         # Reinforcement update logic (softmax bandit, epsilon exploration)
│
├── defender_engine/        # Defender AI — threshold optimization, budget allocation
│   ├── agent.py            # DefenderAgent: cost-based threshold grid search, policy updates
│   └── policies.py         # Threshold tuning primitives
│
├── core_engine/            # Orchestration and closed-loop execution
│   ├── loop.py             # Main adversarial loop: simulate → detect → update → repeat
│   └── orchestrator.py     # End-to-end pipeline: train, evaluate, plot, stress-test
│
├── simulation/             # Network simulation layer
│   ├── network.py          # Multi-timestep peer network with honest/attacker populations
│   ├── environment.py      # Adversarial scenario engine (surge, stealth, coordinated)
│   └── geo.py              # GPS and location modeling
│
├── evaluation/             # Metrics, economic modeling, visualization
│   ├── metrics.py          # Detection timing, false positive rates, consistency
│   ├── economic.py         # Fraud profit, reward loss, cost modeling
│   └── visualization.py    # All plots (strategy mix, threshold, equilibrium, leakage)
│
├── features/               # Feature engineering
│   └── ...                 # 18 behavioral features × 3 temporal aggregations = 54 model inputs
│
├── modeling/               # ML model training and benchmarking
│   └── ...                 # RandomForest, XGBoost, rolling-window RF, heuristic baseline
│
├── experiments/            # Experiment CLI and benchmark runner
│   └── cli.py              # simulate / train / evaluate / stress-test commands
│
├── api/                    # FastAPI scoring endpoint
│   └── ...                 # POST /score-node → fraud_score, trend, confidence, reasons
│
├── run_demo.py             # 🎯 Start here — live adversarial demo with narrative output
├── streamlit_app.py        # Interactive web UI
└── requirements.txt
```

---

##  Future Work

- **Real-world data integration** — connect to live DePIN network telemetry feeds
- **On-chain deployment** — emit fraud scores directly to smart contracts
- **Federated defense** — run defender agents across multiple independent network operators
- **Multi-defender competition** — model competing defenders with different cost structures
- **Regulatory reporting** — auto-generate audit trails for compliance use cases

---

##  Quick Reference

| Command | What it does |
|---|---|
| `python run_demo.py` | Run live adversarial demo |
| `python run_demo.py --scenario spike` | Run burst-attack scenario |
| `python -m core_engine.orchestrator` | Full train + evaluate pipeline |
| `python -m experiments.cli benchmark` | Benchmark RF vs XGBoost vs heuristic |
| `uvicorn api:app --port 8000` | Start real-time scoring API |
| `python -m unittest -v tests/test_pipeline.py` | Run all tests |

## Upgrade highlights (realism + product readiness)

- Scenario difficulty engine: `difficulty_level=easy|medium|hard|extreme`
- Bounded-rational smart attackers with noisy/delayed model-feedback awareness
- Strategy marketplace primitives (`attacker_engine/strategies.py`) with strategy mutation
- Semi-supervised labeling realism:
  - delayed labels
  - partially labeled data (`is_labeled`, `effective_label`)
  - permanent ambiguity flags (`is_ambiguous`)
- Correlated and latent feature noise for harder separation
- Benchmark mode with:
  - RandomForest
  - XGBoost
  - heuristic baseline
  - rolling-window RF (sequence-aware approximation)
- Distribution-shift evaluation: train on medium difficulty, evaluate on hard difficulty
- Product/system cost metrics:
  - false-positive cost
  - false-negative cost
  - detection latency penalty
  - economic delay loss
- Explainability/API upgrades:
  - top contributing features
  - confidence reasoning
  - uncertainty sources
- Product bridge package:
  - `product_bridge/simulation`
  - `product_bridge/features`
  - `product_bridge/modeling`
  - `product_bridge/evaluation`
  - `product_bridge/api`
  - `product_bridge/experiments`
- Closed-loop adversarial intelligence modules:
  - `core_engine/`
  - `attacker_engine/`
  - `defender_engine/`
  - `experiments/`
- Strategic game-theoretic loop upgrades:
  - forward-looking attacker utility simulation with limited foresight
  - defender threshold cost optimization under budget constraints
  - iterative alternating best-response interaction dynamics
  - equilibrium detection (`detect_equilibrium`)
  - cross-run memory persistence (`memory/attacker_memory.json`, `memory/defender_memory.json`)
  - strategic adversarial scenarios + system-reactive environment updates
  - evolution plots (strategy mix, threshold, cost, fraud leakage, equilibrium)

## Closed-loop adversarial intelligence architecture

```text
core_engine (simulation + orchestration)
├── attacker_engine
│   ├── strategy marketplace
│   ├── reinforcement reward history
│   └── softmax strategy-mix adaptation
├── defender_engine
│   ├── budgeted top-K investigations
│   ├── adaptive threshold policy
│   └── threshold arm exploration
├── core_engine/loop.py
│   ├── simulate -> train/update -> predict -> defend -> reward/update
│   ├── hard-negative mining
│   └── curriculum + failure-driven augmentation
├── api (api.py scoring endpoint)
│   └── partial observability mode (dropout, delayed/conflicting evidence)
└── experiments
    ├── benchmark-as-a-service
    └── CLI: simulate/train/evaluate/stress-test
```

## Closed-loop lifecycle

1. Generate adversarial simulation at curriculum-selected difficulty.  
2. Train fraud model and deploy into next simulation round.  
3. Collect false positives/false negatives and delayed detections.  
4. Mine hard examples and upweight them for retraining.  
5. Use failure categories to augment next-round scenarios:
   - mimicry failure -> more camouflage/perfect-mimic pressure
   - noise failure -> stronger honest anomaly/noise injection
   - delayed detection -> more temporal drift and delayed observations
6. Repeat until metrics stabilize under adversarial pressure.

## CLI modes

```bash
python -m experiments.cli run_simulation
python -m experiments.cli run_closed_loop --rounds 4 --output-dir closed_loop_plots
python -m experiments.cli benchmark
python -m experiments.cli train_model
```

## Project goals

This repository simulates a decentralized peer network with honest and attacker nodes, extracts robust behavioral features, trains fraud detectors, evaluates detection and economic impact, and serves a real-time scoring endpoint.

Core goals:
- Stress-test fraud detection against adaptive and scenario-driven attackers.
- Measure detection quality over time (not only static accuracy).
- Quantify downstream fraud economics.
- Expose explainable online scoring for new observations.

## Repository structure

- `simulation.py`  
  Builds multi-timestep network simulations with:
  - honest, naive attacker, and smart attacker populations
  - adaptive smart-attacker state updates (feature-targeted evasion + stealth budgets)
  - optional system-level attack injections:
    - regional attack
    - sybil swarm
    - cascading trust failure
- `features.py`  
  Extracts per-snapshot and temporal node features, then aggregates into model-ready columns.
- `utils/constants.py`  
  Defines base features, aggregation suffixes (`mean/std/last`), and full `FEATURE_COLUMNS`.
- `modeling.py`  
  Trains and compares `RandomForestClassifier` and `XGBClassifier`, chooses a threshold, persists model bundle (`model.pkl`), and returns evaluation artifacts.
- `evaluation/metrics.py`  
  Computes per-node detection timing/consistency and summary false-positive/delay metrics.
- `evaluation/economic.py`  
  Simulates reward outcomes with region and connectivity multipliers, including high-value region attack success.
- `evaluation/visualization.py`  
  Produces figures: network view, fraud score distribution, ROC curve, detection over time, reward distribution.
- `core_engine/orchestrator.py`  
  End-to-end orchestration from baseline simulation to adaptive simulation, metrics, artifacts, and plots.
- `api.py`  
  FastAPI service with `POST /score-node` for online fraud scoring, trend, confidence, and top reason signals.
- `tests/test_pipeline.py`  
  Integration-style tests covering simulation shape, feature schema, model training, and API scoring behavior.

## Requirements

- Python 3.10+ recommended
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

### 1) Run the full pipeline

```bash
python -m core_engine.orchestrator
```

Main artifacts:
- `model.pkl` (selected model + threshold + feature metadata)
- `network.png`
- `fraud_score_distribution.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `calibration_curve.png`
- `performance_vs_noise.png`
- `detection_over_time.png`
- `reward_distribution.png`
- console summaries for model metrics, detection metrics, economic metrics, scenario metrics, and stress tests
- additional summaries for:
  - benchmark comparison (`artifacts.benchmark`)
  - robustness suite (`artifacts.robustness`)
  - failure analysis (`artifacts.failure_analysis`)
  - system cost metrics (`artifacts.system_costs`)

### Difficulty-driven simulation example

```python
from simulation import build_network_simulation

sim = build_network_simulation(
    total_nodes=300,
    time_steps=24,
    difficulty_level="hard",
    seed=7,
)
```

### Benchmark example

```python
from features import extract_node_features
from modeling import benchmark_fraud_models
from simulation import build_network_simulation

train_sim = build_network_simulation(seed=11, difficulty_level="medium")
test_sim = build_network_simulation(seed=12, difficulty_level="extreme")
benchmark_df = benchmark_fraud_models(
    train_dataset=extract_node_features(train_sim),
    test_dataset=extract_node_features(test_sim),
    random_state=11,
)
print(benchmark_df)
```

### 2) Run API scoring service

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 3) Run tests

```bash
python -m unittest -v tests/test_pipeline.py
```

## End-to-end workflow

1. **Baseline simulation** (`build_network_simulation`) generates time-varying latency and peer-graph snapshots.  
2. **Feature extraction** builds temporal anomaly features and model-ready aggregates.  
3. **Modeling** trains RF + XGBoost, selects best model and threshold, saves `model.pkl`.  
4. **Adaptive simulation** reruns with model-in-the-loop smart-attacker adaptation.  
5. **Detection metrics** summarize detection speed/consistency and false positives over time.  
6. **Economic simulation** estimates fraud profit and reward loss to fraud.  
7. **Visualization** writes diagnostic plots for model and system behavior.  

## Key simulation controls

`build_network_simulation(...)` supports key controls for experiment design:

- population and ratio controls:
  - `total_nodes`
  - `honest_ratio`, `naive_attacker_ratio`, `smart_attacker_ratio`
- noise/observability controls:
  - `measurement_error_std`
  - `missing_latency_rate`
  - `packet_loss_rate`
  - `partial_visibility_rate`
  - `noise_level`
  - `visibility_level`
- temporal controls:
  - `time_steps`
  - `seed`
- adaptive attacker controls:
  - `adaptation_base`, `adaptation_growth`
  - `attacker_sophistication`
  - `model_for_adaptation`, `fraud_score_threshold`, `feature_columns`
  - `smart_strategy_mix`, `smart_target_top_k`
  - `label_noise_rate`
- scenario toggle:
  - `enable_system_attacks`

## Feature schema

The model uses 18 base behavioral features (see `constants.BASE_FEATURE_COLUMNS`) expanded into:
- `<feature>_mean`
- `<feature>_std`
- `<feature>_last`

The aggregation is intentionally non-naive: recent-weighted partial history + rolling instability summaries
are mapped into these suffixes to preserve temporal instability while keeping a fixed model schema.

Total model inputs:
- `len(FEATURE_COLUMNS) = 18 * 3 = 54`

Feature meanings are available via:
- `features.feature_meanings_dataframe()`

## API contract

### Endpoint

- `POST /score-node`

### Input highlights (`ScoreNodeRequest`)

- `latencies` (required): current observed forward latencies
- `peers` (required): peer identifiers for current observation
- `claimed_location` (required): `{lat, lon}`
- optional context:
  - `inferred_location`
  - `reverse_latencies`
  - `peer_claimed_locations`
  - `clustering_coefficient`
  - `reciprocity_score`
  - `past_fraud_scores`
  - `history_latencies`

### Output (`ScoreNodeResponse`)

- `fraud_score` (`0..1`)
- `uncertainty` (`0..1`, ensemble disagreement)
- `uncertainty_score` (`0..1`, ensemble disagreement)
- `risk_label` (`high_risk` / `low_risk`)
- `trend` (`increasing_risk` / `decreasing_risk` / `stable_risk`)
- `confidence_score` (`0..1`)
- `reasons` (top explainability signals)

### API model configuration

- `MODEL_PATH` env var: path to saved model bundle (defaults to `model.pkl`)
- `FRAUD_SCORE_THRESHOLD` env var: optional runtime threshold override

## Metrics produced

- **Model metrics**: accuracy, precision, recall, ROC-AUC, FPR/FNR, confusion matrix
- **Detection metrics**:
  - per node: `first_detection_timestep`, `detection_consistency`, `detection_delay`, `was_detected`
  - summary: average detection delay, % attackers detected within N steps, average false positives over time
- **Economic metrics**:
  - `total_fraud_profit`
  - `fraud_reduction_after_detection`
  - `attacker_profit_vs_honest_profit_ratio`
  - `pct_reward_lost_to_fraud`
  - `fraud_concentration_by_region`
  - `high_value_attack_success_rate`

## Reproducibility notes

- Use explicit `seed` values in simulation and pipeline runs for deterministic experiment comparison.
- Scenario activations are stochastic; compare across multiple seeds for robust conclusions.

## Troubleshooting

- **`ModuleNotFoundError` during tests/run**  
  Ensure dependencies are installed from `requirements.txt`.
- **API returns model file error**  
  Run `python -m core_engine.orchestrator` first (creates `model.pkl`), or set `MODEL_PATH` to a valid trained bundle.
