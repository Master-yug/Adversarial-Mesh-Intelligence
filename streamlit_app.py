#!/usr/bin/env python3
"""
streamlit_app.py — Interactive Adversarial Battle Simulation Dashboard
========================================================================
A Streamlit dashboard that lets you tune attacker strength, defender
budget, and noise level then watch the battle play out in real time.

Usage
-----
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from attacker_engine.agent import AttackerAgent
from attacker_engine.strategies import STRATEGY_MARKETPLACE
from core_engine.loop import (
    AdversarialFraudEnvironment,
    ClosedLoopIteration,
    analyze_system_dynamics,
)
from defender_engine.agent import DefenderAgent
from evaluation.visualization import (
    plot_cost_vs_time,
    plot_defender_threshold_over_time,
    plot_fraud_leakage_vs_time,
    plot_strategy_stacked_area,
)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚔ Adversarial Fraud Battle",
    page_icon="⚔",
    layout="wide",
)

STRATEGY_DISPLAY: Dict[str, str] = {
    "low_and_slow":   "Low-and-Slow",
    "burst_attack":   "Burst Attack",
    "camouflage":     "Camouflage",
    "perfect_mimic":  "Perfect Mimic",
    "slow_drift":     "Slow Drift",
    "decoy_attacker": "Decoy Attacker",
    "mixed_cluster":  "Mixed Cluster",
}

SCENARIO_SCHEDULES: Dict[str, List[str]] = {
    "Normal (Balanced)": [
        "steady_state_low_level_fraud",
        "coordinated_attack_campaign",
        "defender_overload",
        "steady_state_low_level_fraud",
        "stealth_long_term_infiltration",
        "coordinated_attack_campaign",
    ],
    "Attack Spike": [
        "sudden_attack_surge",
        "sudden_attack_surge",
        "coordinated_attack_campaign",
        "sudden_attack_surge",
        "defender_overload",
        "steady_state_low_level_fraud",
    ],
    "Stealth": [
        "stealth_long_term_infiltration",
        "stealth_long_term_infiltration",
        "steady_state_low_level_fraud",
        "stealth_long_term_infiltration",
        "stealth_long_term_infiltration",
        "steady_state_low_level_fraud",
    ],
    "Coordinated": [
        "coordinated_attack_campaign",
        "defender_overload",
        "coordinated_attack_campaign",
        "defender_overload",
        "coordinated_attack_campaign",
        "sudden_attack_surge",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
def _run_baseline(nodes: int, seed: int, schedule: List[str]) -> Optional[float]:
    """1-step simulation with a fully passive defender to establish baseline."""
    try:
        env = AdversarialFraudEnvironment(
            iterations=1,
            seed=seed + 9999,
            total_nodes=nodes,
            time_steps=16,
            attacker_memory_path=Path("/tmp/st_baseline_attacker.json"),
            defender_memory_path=Path("/tmp/st_baseline_defender.json"),
            scenario_schedule=schedule[:1],
        )
        pa = AttackerAgent(temperature=0.25, epsilon=0.08, use_ucb_selection=True)
        pd_ = DefenderAgent(threshold=0.999, budget_ratio=0.0, min_budget=0)
        outcome = env.step(iteration=0, attacker_agent=pa, defender_agent=pd_)
        return float(outcome.result.metrics.get("fraud_leakage_pct", 50.0))
    except Exception:
        return None


def _run_simulation(
    nodes: int,
    iterations: int,
    seed: int,
    attacker_temperature: float,
    defender_budget: float,
    schedule: List[str],
    progress_bar,
    status_text,
) -> tuple[List[ClosedLoopIteration], Dict, Optional[float], Dict]:
    """Run the full adversarial simulation and return rounds + analysis."""
    env = AdversarialFraudEnvironment(
        iterations=iterations,
        seed=seed,
        total_nodes=nodes,
        time_steps=16,
        attacker_memory_path=Path("/tmp/st_attacker.json"),
        defender_memory_path=Path("/tmp/st_defender.json"),
        scenario_schedule=schedule[:iterations],
    )
    attacker = AttackerAgent(temperature=attacker_temperature, epsilon=0.08, use_ucb_selection=True)
    defender = DefenderAgent(threshold=0.5, budget_ratio=defender_budget, min_budget=3)

    rounds: List[ClosedLoopIteration] = []
    for i in range(iterations):
        status_text.text(f"⚔ Round {i + 1}/{iterations} — battle in progress…")
        outcome = env.step(iteration=i, attacker_agent=attacker, defender_agent=defender)
        rounds.append(outcome.result)
        progress_bar.progress((i + 1) / iterations)

    analysis = analyze_system_dynamics(env.history, rounds=rounds)
    history  = env.history

    # Baseline
    status_text.text("📊 Running baseline (no-defender) simulation…")
    baseline = _run_baseline(nodes=nodes, seed=seed, schedule=schedule)

    return rounds, analysis, baseline, history


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("⚔  Adversarial Fraud Battle Simulation")
    st.markdown(
        "_A live battle between attackers and defenders in a decentralised network.  "
        "Tune the sliders and watch the war unfold._"
    )
    st.divider()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️  Battle Configuration")

        scenario_name = st.selectbox(
            "Scenario",
            list(SCENARIO_SCHEDULES.keys()),
            index=0,
            help="Choose the attack scenario to simulate",
        )
        nodes = st.slider(
            "Network size (nodes)",
            min_value=50, max_value=500, value=150, step=50,
            help="Number of nodes in the simulated network",
        )
        iterations = st.slider(
            "Battle rounds",
            min_value=2, max_value=10, value=4, step=1,
            help="How many rounds the simulation runs",
        )
        attacker_strength = st.slider(
            "🔴 Attacker strength (temperature)",
            min_value=0.05, max_value=1.0, value=0.25, step=0.05,
            help="Higher = more exploratory/aggressive attacker strategies",
        )
        defender_budget = st.slider(
            "🔵 Defender budget ratio",
            min_value=0.02, max_value=0.30, value=0.07, step=0.01,
            help="Fraction of the network the defender can investigate per round",
        )
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

        run_btn = st.button("🚀  Run Battle", type="primary", use_container_width=True)

    # ── Main area ─────────────────────────────────────────────────────────────
    if not run_btn:
        st.info(
            "👈  Configure the battle parameters in the sidebar, then click **Run Battle**."
        )
        return

    schedule = SCENARIO_SCHEDULES[scenario_name]
    while len(schedule) < iterations:
        schedule = schedule + schedule
    schedule = schedule[:iterations]

    progress_bar = st.progress(0)
    status_text  = st.empty()

    with st.spinner("Initialising battle…"):
        rounds, analysis, baseline, history = _run_simulation(
            nodes=nodes,
            iterations=iterations,
            seed=int(seed),
            attacker_temperature=attacker_strength,
            defender_budget=defender_budget,
            schedule=schedule,
            progress_bar=progress_bar,
            status_text=status_text,
        )

    progress_bar.empty()
    status_text.empty()

    # ── Metrics summary ───────────────────────────────────────────────────────
    mean_leakage  = float(analysis.get("mean_fraud_leakage_pct", 0.0))
    mean_cost     = float(analysis.get("mean_system_cost", 0.0))
    eq_reached    = bool(analysis.get("equilibrium_reached", False))
    dominant      = str(analysis.get("dominant_strategy", "unknown"))
    efficiency    = float(analysis.get("system_efficiency", 0.0))

    if baseline is not None:
        reduction    = max(0.0, float(baseline) - mean_leakage)
        prevented    = min(100.0, reduction / max(float(baseline), 1e-9) * 100.0)
    else:
        prevented = float(analysis.get("fraud_prevented_pct", 0.0))

    st.subheader("📊 Battle Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔥 Mean Fraud Leakage", f"{mean_leakage:.1f}%",
              delta=f"-{prevented:.1f}% vs no defender" if baseline else None,
              delta_color="inverse")
    c2.metric("💰 Mean System Cost",   f"{mean_cost:.1f}")
    c3.metric("🛡 Fraud Prevented",    f"{prevented:.1f}%")
    c4.metric("⚖ Equilibrium",         "✅ Yes" if eq_reached else "⚡ No")
    c5.metric("🎯 Dominant Strategy",  STRATEGY_DISPLAY.get(dominant, dominant))

    st.divider()

    # ── Before / After banner ─────────────────────────────────────────────────
    st.subheader("⚔ BEFORE vs AFTER — Fraud Prevention Impact")
    ba_col1, ba_col2 = st.columns(2)
    with ba_col1:
        without_val = float(baseline) if baseline is not None else mean_leakage + 10
        st.metric(
            "❌ Without Anti-Spoofing System",
            f"{without_val:.1f}%",
            help="Fraud leakage measured with a fully passive (no-op) defender",
        )
        st.progress(min(1.0, without_val / 100.0))
    with ba_col2:
        st.metric(
            "✅ With Anti-Spoofing System",
            f"{mean_leakage:.1f}%",
            help="Mean fraud leakage measured across all simulation rounds",
        )
        st.progress(min(1.0, mean_leakage / 100.0))

    source = "(actual no-defender simulation)" if baseline is not None else "(estimated)"
    st.success(f"🛡  **System prevented {prevented:.1f}% of fraud losses** {source}")

    st.divider()

    # ── Narrative timeline ────────────────────────────────────────────────────
    st.subheader("📜 Live Battle Timeline")
    leakage_hist = [float(v) for v in history.get("fraud_leakage", [])]
    prev_leakage = 0.0

    for i, it in enumerate(rounds):
        leakage    = float(it.metrics.get("fraud_leakage_pct", 0.0))
        cost       = float(it.metrics.get("total_system_cost", 0.0))
        threshold  = float(it.defender_threshold)
        flagged    = len(it.selected_nodes)
        delta      = leakage - prev_leakage

        events: List[str] = []
        if i > 0 and delta > 15:
            events.append(f"⚠️ Attack surge detected (+{delta:.1f}%)")
        if i > 0 and delta < -10:
            events.append(f"✅ System stabilizing ({delta:.1f}%)")
        dominant_s = max(it.attacker_strategy_mix.items(), key=lambda x: x[1], default=("none", 0))
        if dominant_s[0] in ("low_and_slow", "slow_drift", "camouflage") and dominant_s[1] > 0.35:
            events.append(f"🕵️ Attackers switch to stealth: {STRATEGY_DISPLAY.get(dominant_s[0], dominant_s[0])}")
        if it.equilibrium_detected:
            events.append("🏁 Equilibrium detected — battle stabilising")
        if leakage > 40:
            events.append(f"🔥 Fraud spike! Leakage at {leakage:.1f}%")

        event_str = "  |  ".join(events) if events else "—"
        with st.expander(
            f"Round {i + 1}/{len(rounds)}  ·  Fraud {leakage:.1f}%  ·  Cost {cost:.1f}  ·  {it.selected_attacker_strategy}",
            expanded=(i == len(rounds) - 1),
        ):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Fraud Leakage",     f"{leakage:.1f}%")
            col_b.metric("System Cost",        f"{cost:.1f}")
            col_c.metric("Defender Threshold", f"{threshold:.3f}")
            if events:
                for ev in events:
                    st.markdown(f"**{ev}**")
            st.caption(f"Scenario: {it.selected_adversarial_scenario}  |  Strategy: {it.selected_attacker_strategy}")

        prev_leakage = leakage

    st.divider()

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.subheader("📈 Visualizations")
    threshold_hist = [float(v) for v in history.get("defender_threshold", [])]
    cost_hist      = [float(v) for v in history.get("system_cost", [])]
    strategy_hist  = history.get("attacker_strategy_distribution", [])

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("**Fraud Leakage Over Time**")
        if leakage_hist:
            fig, ax = plot_fraud_leakage_vs_time(leakage_hist, baseline_leakage=baseline, output_path=None)
            ax.set_title("Fraud Leakage", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Attacker Strategy Evolution**")
        if strategy_hist:
            fig, ax = plot_strategy_stacked_area(strategy_hist, output_path=None)
            ax.set_title("Strategy Mix", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

    with plot_col2:
        st.markdown("**Defender Threshold Over Time**")
        if threshold_hist:
            fig, ax = plot_defender_threshold_over_time(threshold_hist, output_path=None)
            ax.set_title("Defender Threshold", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**System Cost Over Time**")
        if cost_hist:
            fig, ax = plot_cost_vs_time(cost_hist, output_path=None)
            ax.set_title("System Cost", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

    # ── Before/After bar chart ────────────────────────────────────────────────
    st.markdown("**Before vs After — Fraud Comparison**")
    without_val = float(baseline) if baseline is not None else mean_leakage + 10
    fig_ba, ax_ba = plt.subplots(figsize=(7, 4))
    bars = ax_ba.bar(
        ["Without Defender\n(Baseline)", "With Defender\n(This Run)"],
        [without_val, mean_leakage],
        color=["#ef4444", "#22c55e"],
        width=0.45,
        edgecolor="white",
        linewidth=1.5,
    )
    ax_ba.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=12, fontweight="bold")
    ax_ba.set_ylabel("Fraud Leakage (%)")
    ax_ba.set_ylim(0, max(without_val, mean_leakage) * 1.5 + 5)
    ax_ba.set_title(f"System prevented {prevented:.1f}% of fraud", fontsize=13, fontweight="bold")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig_ba.tight_layout()
    st.pyplot(fig_ba)
    plt.close(fig_ba)

    st.divider()

    # ── Final outcome ─────────────────────────────────────────────────────────
    st.subheader("🏁 Final Outcome")
    if eq_reached:
        eq_type = str(analysis.get("equilibrium_type", "stable"))
        st.success(
            f"✅ **Equilibrium reached** — {eq_type.replace('_', ' ').title()}.  "
            f"Attacker and defender locked into stable strategies."
        )
    else:
        st.warning(
            "⚡ **No equilibrium reached** — the arms race is ongoing.  "
            "The battle continues beyond this simulation window."
        )

    st.info(
        f"🛡  **System prevented {prevented:.1f}% of fraud losses** over {len(rounds)} rounds.  "
        f"Dominant attacker strategy: **{STRATEGY_DISPLAY.get(dominant, dominant)}** "
        f"({float(analysis.get('dominance_share', 0)):.0%} share)."
    )


if __name__ == "__main__":
    main()
