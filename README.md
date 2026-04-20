# Adversarial Mesh Intelligence

Adversarial decentralized network anti-spoofing simulation and detection pipeline.

## What is included

- `simulation.py`: Configurable multi-round simulation with honest, naive attacker, and smart attacker nodes; feature-targeted smart adaptation with stealth budget; and injected network-wide attack scenarios (regional attack, sybil swarm, cascading trust failure).
- `features.py`: Robust adversarial feature extraction from noisy/partial observations, including temporal anomaly features (latency spikes, trend slope, volatility, sudden changes, burst activity) and feature-meaning table.
- `modeling.py`: Trains and compares `RandomForestClassifier` and `XGBClassifier`, reports accuracy/precision/recall/ROC-AUC/confusion matrix, analyzes FP/FN, and saves best `model.pkl`.
- `detection_metrics.py`: Per-node detection timing/consistency metrics and summary detection stats over time.
- `economic.py`: Dynamic regional and connectivity-aware reward simulation with fraud concentration and high-value attack success metrics.
- `visualization.py`: Network plot + fraud score distribution + ROC curve + detection-over-time + reward-distribution figures.
- `api.py`: FastAPI service with stateful `POST /score-node` returning `fraud_score`, `risk_label`, risk `trend`, `confidence_score`, and explainable `reasons`.
- `pipeline.py`: End-to-end baseline training + model-in-the-loop adaptive attacker simulation + metrics + plots.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run end-to-end pipeline

```bash
python pipeline.py
```

Pipeline outputs include:
- Node metadata (`real_location`, `claimed_location`, labels)
- Time-series latency and graph behavior (`time_steps`)
- Aggregated model-ready features (`pandas.DataFrame`)
- Model comparison summary and feature-importance comparison
- Detection-over-time metrics (`first_detection_timestep`, `detection_consistency`, `detection_delay`)
- Economic impact metrics (`total_fraud_profit`, `fraud_reduction_after_detection`, reward-loss%)
- Saved best model (`model.pkl`)

## Run API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Run tests

```bash
python -m unittest -v tests/test_pipeline.py
```
