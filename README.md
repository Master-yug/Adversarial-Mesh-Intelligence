# antispoof

Adversarial decentralized-network anti-spoofing simulation, ML detection, evaluation, and online scoring API.

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
