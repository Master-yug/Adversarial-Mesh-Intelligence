# Adversarial-Mesh-Intelligence
## One-Page Summary

---

### The Problem

Decentralized networks are losing billions to fraud.

GPS spoofing, Sybil attacks, and fake node injection let bad actors claim rewards they never earned. Every dollar paid to a fraudster is stolen from honest participants.

Current detection systems use static rules and one-time-trained models. Attackers figure them out in days — then exploit them indefinitely. The arms race has no end.

---

### The Solution

**Adversarial-Mesh-Intelligence** is a self-learning fraud detection engine that treats the attacker as an intelligent opponent — not just a pattern to match.

Two AI agents compete continuously:

- 🔴 **The Attacker** — chooses from 7 fraud strategies, simulates the defender's reaction before acting, and updates its strategy mix every round based on what worked
- 🔵 **The Defender** — optimizes its detection threshold and investigation budget every round by minimizing a real cost function: missed fraud + false alarms + investigation overhead

The system runs in closed loops until both agents stabilize — reaching **Nash equilibrium**. That equilibrium is the network's true security baseline.

---

### Why It Matters

Most fraud systems are built around yesterday's attacks. Ours is built around tomorrow's.

By modeling the attacker as a learning agent, we can answer questions that static systems can't:

- What happens when attackers shift from burst attacks to stealth infiltration?
- What's the minimum detection budget to stay below 10% fraud leakage?
- When does the defender's adaptation converge — and is it holding?

This is fraud detection as a **strategy problem**, not a labeling problem.

---

### Key Innovations

| Innovation | Impact |
|---|---|
| **Game-theoretic attacker model** | Defender is tested against an opponent that anticipates and adapts |
| **Forward simulation with foresight** | Attacker looks 1–2 steps ahead before choosing strategies |
| **Cost-optimized defender** | Threshold and budget jointly minimized against a real cost function |
| **Equilibrium detection** | System reports its own convergence — no black-box guarantees |
| **7-strategy marketplace** | Covers burst, stealth, camouflage, cluster, and mimic attack classes |
| **Cross-run memory** | Strategic learning persists across sessions |
| **54 behavioral features** | Temporal, spatial, trust, and graph signals across 18 dimensions |
| **Real-time scoring API** | Production-ready fraud scores with confidence, trend, and explainability |

---

### What the System Produces

Running a single demo generates:

- Per-round fraud leakage, detection rate, and system cost
- Strategy mix evolution (which fraud tactics dominate)
- Defender threshold adaptation trajectory
- Equilibrium detection signal
- Economic impact: total fraud profit, % rewards lost to fraud, regional fraud concentration
- Trained fraud model deployable via REST API

---

### Real-World Fit

This system is designed for:

- **DePIN networks** — GPS-based reward systems (mobility, energy, wireless)
- **Web3 peer networks** — Sybil-resistant node verification
- **Decentralized oracles** — Trust scoring for on-chain data providers
- **Fraud analytics platforms** — Adaptive detection for any adversarial environment

The architecture is modular. The simulation layer, ML layer, attacker engine, and defender engine are all independently deployable.

---

### Run It

```bash
pip install -r requirements.txt
python run_demo.py --scenario spike
```

Output: live terminal narrative + 5 saved plots + full metrics summary.

---

*Built for the real arms race — not the last one.*
