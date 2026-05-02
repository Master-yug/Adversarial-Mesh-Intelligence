#  Demo Guide — Adversarial-Mesh-Intelligence

## Before You Start

Install dependencies (one-time setup):

```bash
python -m pip install -r requirements.txt
```

---

## Running the Demo

The main demo is a **live, round-by-round simulation** of attackers and defenders competing in a decentralized network. It prints a narrative to your terminal and saves plots when done.

```bash
python run_demo.py [--scenario SCENARIO] [--nodes N] [--seed N] [--output-dir DIR]
```

---

## Scenarios

Choose a scenario based on what you want to demonstrate:

### `normal` — Balanced Environment (default)

```bash
python run_demo.py --scenario normal
```

**What happens:** The simulation runs through a mix of attack types — steady-state fraud, a coordinated campaign, a defender overload, and a stealth infiltration. The defender has time to stabilize.

**Best for:** Showing the full attacker-defender lifecycle, including equilibrium detection.

**What to watch:**
- Strategy mix gradually converges as unprofitable strategies are abandoned
- Defender threshold settles into a stable range
- Fraud leakage trends downward over rounds

---

### `spike` — Sudden Attack Surge

```bash
python run_demo.py --scenario spike
```

**What happens:** Three consecutive sudden-surge attacks hit the network before the defender has adapted. High fraud leakage early, followed by an aggressive defender response.

**Best for:** Showing how the system handles shock events and recovers.

**What to watch:**
- Early rounds: fraud leakage spikes, system cost is high
- Defender rapidly lowers its threshold (flags more nodes aggressively)
- Later rounds: system cost drops as defender catches up
- Attacker shifts away from burst strategies toward stealthier options

---

### `stealth` — Long-Term Infiltration

```bash
python run_demo.py --scenario stealth
```

**What happens:** Attackers use `low_and_slow`, `perfect_mimic`, and `slow_drift` tactics — strategies designed to look like honest behavior. The defender takes more rounds to detect them.

**Best for:** Showing the hardest detection challenge and the importance of temporal features.

**What to watch:**
- Fraud leakage stays elevated longer (stealth attacks are genuinely hard to catch)
- Model uncertainty is higher — the defender isn't sure which nodes are fraudulent
- Detection consistency scores are lower than other scenarios
- Eventually, the accumulation of behavioral drift signals exposes the attackers

---

### `coordinated` — Cluster Attack Campaign

```bash
python run_demo.py --scenario coordinated
```

**What happens:** Attacker clusters coordinate to overload the defender's investigation budget. The defender can only investigate a fixed % of nodes per round, so coordinated attacks can exceed capacity.

**Best for:** Showing the budget constraint challenge and how the system adapts its budget allocation.

**What to watch:**
- Early rounds: many attackers slip through because the investigation budget is exceeded
- Defender increases its budget ratio over time
- Selection strategy shifts to prioritize highest-risk nodes
- System cost reflects the trade-off between budget and missed detections

---

## What the Terminal Output Shows

Each round prints:

```
═══════════════════════════════════════════════════════
  ROUND 3 — Coordinated Campaign
  Scenario: coordinated_attack_campaign
═══════════════════════════════════════════════════════

  🔴 ATTACKER  Strategy: perfect_mimic  (P=0.34)
               Expected utility:  0.21
               Forward simulation gain: 0.48

  🔵 DEFENDER  Threshold:  0.61   Budget: 7%
               Selection: risk_adjusted

  📊 RESULTS
               Fraud leakage:    18.3%
               False positive:    4.1%
               System cost:      2.84
               Detected attackers: 81.7%

  ⚖️  Equilibrium: Not yet reached (round 3/6)
```

**Key metrics to interpret:**

| Metric | What it means | Good value |
|---|---|---|
| Fraud leakage | % of attackers not caught | Lower is better |
| False positive rate | % of honest nodes wrongly flagged | Lower is better |
| System cost | Combined cost (FP + FN + investigation) | Lower is better |
| Detected attackers | % of real fraudsters caught | Higher is better |
| Threshold | Defender's sensitivity cutoff | Watch it adapt |

---

## Output Plots

After the demo completes, plots are saved to `demo_output/` (or your `--output-dir`):

### `strategy_distribution.png`
A stacked area chart showing how the attacker's strategy mix evolves over rounds. Watch which strategies grow (effective) and which shrink (too costly or easily detected).

**Insight:** Convergence in this chart signals the attacker has found its optimal mix.

### `defender_threshold.png`
A line chart of the defender's threshold over time. A rising threshold means fewer flags (defender became more conservative). A falling threshold means more flags (defender became more aggressive).

**Insight:** Oscillation then stabilization is the expected pattern before equilibrium.

### `fraud_leakage.png`
How much fraud is slipping through each round. Should trend downward as the defender adapts.

**Insight:** Stealth scenarios will show slower decline. Spike scenarios will show a fast initial spike then rapid drop.

### `system_cost.png`
The total defense cost per round. A U-shaped trajectory is common: high early (learning), low in the middle (adapted), potentially rising slightly at equilibrium if the attacker finds a residual profitable strategy.

### `equilibrium.png`
A binary signal showing when equilibrium was first detected. The earlier this fires, the faster the system stabilized.

---

## Demo with More Options

```bash
# Larger network (slower but more realistic)
python run_demo.py --scenario coordinated --nodes 500

# Reproducible run with fixed seed
python run_demo.py --scenario spike --seed 123

# Quiet mode (metrics only, no narrative text)
python run_demo.py --quiet

# No ANSI color (for logging/capture)
python run_demo.py --no-color

# Custom output directory
python run_demo.py --output-dir /tmp/my_demo
```

---

## Interpreting a Single Round (Step-by-Step)

Let's walk through what happens in one round of the `spike` scenario:

1. **Network is generated** — 200 nodes are created. ~70% honest, ~30% attackers. The `sudden_attack_surge` scenario is applied: extra noise, higher attacker density, burst attack injection.

2. **Features are extracted** — 54 behavioral signals per node are computed from the time-series observations.

3. **ML model scores every node** — Each node gets a fraud score from 0 to 1. Scores above the defender's threshold (e.g., 0.61) are candidates for investigation.

4. **Attacker chooses strategy** — The attacker's forward simulation finds that `burst_attack` has the highest expected utility given the current defender state. It selects it (with some epsilon-exploration randomness).

5. **Defender investigates** — The defender selects the top-K nodes above the threshold, capped by its budget (7% of nodes = 14 nodes). It prioritizes using `risk_adjusted` scoring.

6. **Results computed** — FP rate, FN rate, fraud leakage, system cost are all measured.

7. **Both agents update** — The attacker adds `burst_attack`'s reward to its history. The defender adjusts its threshold based on the FP/FN signal.

8. **Next round starts** with the updated state.

---

## Full Pipeline Demo

For judges who want to see the complete training and evaluation pipeline:

```bash
# Train model + run evaluation + generate all artifacts
python -m core_engine.orchestrator
```

This takes ~1–2 minutes and produces:
- `model.pkl` — trained fraud detector
- Multiple evaluation plots
- Console summary of all metrics (model accuracy, detection speed, economic impact, stress test results)

---

## API Demo

To show real-time scoring:

```bash
# Start the API
uvicorn api:app --host 0.0.0.0 --port 8000

# Score a node (in another terminal)
curl -X POST http://localhost:8000/score-node \
  -H "Content-Type: application/json" \
  -d '{
    "latencies": [12.3, 45.1, 8.9, 200.4],
    "peers": ["node_a", "node_b", "node_c"],
    "claimed_location": {"lat": 37.77, "lon": -122.41}
  }'
```

Response:
```json
{
  "fraud_score": 0.82,
  "risk_label": "high_risk",
  "trend": "increasing_risk",
  "confidence_score": 0.79,
  "uncertainty_score": 0.21,
  "reasons": ["latency_variance_high", "gps_drift_detected", "peer_cluster_anomaly"]
}
```
