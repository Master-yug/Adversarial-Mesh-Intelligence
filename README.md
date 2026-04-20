# antispoof

Adversarial decentralized-network anti-spoofing simulation, ML detection, evaluation, and online scoring API.

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
- `constants.py`  
  Defines base features, aggregation suffixes (`mean/std/last`), and full `FEATURE_COLUMNS`.
- `modeling.py`  
  Trains and compares `RandomForestClassifier` and `XGBClassifier`, chooses a threshold, persists model bundle (`model.pkl`), and returns evaluation artifacts.
- `detection_metrics.py`  
  Computes per-node detection timing/consistency and summary false-positive/delay metrics.
- `economic.py`  
  Simulates reward outcomes with region and connectivity multipliers, including high-value region attack success.
- `visualization.py`  
  Produces figures: network view, fraud score distribution, ROC curve, detection over time, reward distribution.
- `pipeline.py`  
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
python pipeline.py
```

Main artifacts:
- `model.pkl` (selected model + threshold + feature metadata)
- `network.png`
- `fraud_score_distribution.png`
- `roc_curve.png`
- `detection_over_time.png`
- `reward_distribution.png`
- console summaries for model metrics, detection metrics, economic metrics, scenario metrics, and stress tests

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
- temporal controls:
  - `time_steps`
  - `seed`
- adaptive attacker controls:
  - `adaptation_base`, `adaptation_growth`
  - `model_for_adaptation`, `fraud_score_threshold`, `feature_columns`
  - `smart_strategy_mix`, `smart_target_top_k`
- scenario toggle:
  - `enable_system_attacks`

## Feature schema

The model uses 17 base behavioral features (see `constants.BASE_FEATURE_COLUMNS`) expanded into:
- `<feature>_mean`
- `<feature>_std`
- `<feature>_last`

Total model inputs:
- `len(FEATURE_COLUMNS) = 17 * 3 = 51`

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
  Run `python pipeline.py` first (creates `model.pkl`), or set `MODEL_PATH` to a valid trained bundle.
