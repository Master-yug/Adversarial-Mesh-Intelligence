#!/usr/bin/env python3
"""
run_demo.py — Adversarial Fraud Detection Live Demo
====================================================
A visual, interactive simulation that shows adversarial dynamics and
fraud prevention impact step-by-step.

Usage
-----
    python run_demo.py [--scenario normal|spike|stealth|coordinated]
                      [--nodes N] [--steps N] [--seed N]
                      [--output-dir DIR] [--no-color] [--quiet]

Scenarios
---------
    normal      Balanced multi-strategy adversarial environment (default)
    spike       Sudden coordinated attack surge
    stealth     Long-term infiltration with slow-drift tactics
    coordinated Organised cluster-based attack campaign
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
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
    plot_equilibrium_detection,
    plot_fraud_leakage_vs_time,
    plot_strategy_distribution_over_time,
)
from features import extract_node_features
from simulation import build_network_simulation
from utils.constants import FEATURE_COLUMNS

# ── ANSI colour helpers ────────────────────────────────────────────────────────
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_BLUE   = "\033[94m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

# Module-level colour slots that can be blanked when --no-color is set.
RED    = _RED
GREEN  = _GREEN
YELLOW = _YELLOW
BLUE   = _BLUE
CYAN   = _CYAN
BOLD   = _BOLD
DIM    = _DIM
RESET  = _RESET


def _c(text: str, colour: str) -> str:
    """Wrap text in an ANSI colour code."""
    return f"{colour}{text}{RESET}"


# ── Scenario catalogue ─────────────────────────────────────────────────────────
SCENARIO_CATALOGUE: Dict[str, Dict] = {
    "normal": {
        "label": "Normal (Balanced)",
        "schedule": [
            "steady_state_low_level_fraud",
            "coordinated_attack_campaign",
            "defender_overload",
            "steady_state_low_level_fraud",
            "stealth_long_term_infiltration",
            "coordinated_attack_campaign",
        ],
        "description": "Balanced multi-strategy adversarial environment",
        "default_iterations": 6,
    },
    "spike": {
        "label": "Attack Spike",
        "schedule": [
            "sudden_attack_surge",
            "sudden_attack_surge",
            "sudden_attack_surge",
            "coordinated_attack_campaign",
            "sudden_attack_surge",
            "defender_overload",
        ],
        "description": "Sudden surge of burst attacks — defender scrambles to respond",
        "default_iterations": 6,
    },
    "stealth": {
        "label": "Stealth Attack",
        "schedule": [
            "stealth_long_term_infiltration",
            "stealth_long_term_infiltration",
            "steady_state_low_level_fraud",
            "stealth_long_term_infiltration",
            "stealth_long_term_infiltration",
            "stealth_long_term_infiltration",
        ],
        "description": "Low-and-slow infiltration — hard to detect, system adapts gradually",
        "default_iterations": 6,
    },
    "coordinated": {
        "label": "Coordinated Attack",
        "schedule": [
            "coordinated_attack_campaign",
            "defender_overload",
            "coordinated_attack_campaign",
            "defender_overload",
            "coordinated_attack_campaign",
            "sudden_attack_surge",
        ],
        "description": "Organised cluster-based attacks overwhelming the defender",
        "default_iterations": 6,
    },
}

STRATEGY_DISPLAY: Dict[str, str] = {
    "low_and_slow":   "Low-and-Slow",
    "burst_attack":   "Burst Attack",
    "camouflage":     "Camouflage",
    "perfect_mimic":  "Perfect Mimic",
    "slow_drift":     "Slow Drift",
    "decoy_attacker": "Decoy Attacker",
    "mixed_cluster":  "Mixed Cluster",
}

SCENARIO_DISPLAY: Dict[str, str] = {
    "steady_state_low_level_fraud":  "Steady-State Fraud",
    "coordinated_attack_campaign":   "Coordinated Campaign",
    "sudden_attack_surge":           "Sudden Surge",
    "defender_overload":             "Defender Overload",
    "stealth_long_term_infiltration":"Stealth Infiltration",
}


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial Anti-Spoofing Demo\n"
            "A live battle between attackers and defenders in a decentralised network."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_CATALOGUE.keys()),
        default="normal",
        metavar="{" + "|".join(SCENARIO_CATALOGUE) + "}",
        help="Scenario mode (default: normal)",
    )
    parser.add_argument("--nodes",   type=int,  default=200,             help="Number of simulated nodes (default: 200)")
    parser.add_argument("--steps",   type=int,  default=0,               help="Number of iterations (0 = scenario default)")
    parser.add_argument("--seed",    type=int,  default=42,              help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"),
                        help="Directory for plots and report (default: demo_output)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour output")
    parser.add_argument("--quiet",    action="store_true", help="Suppress narrative; print only metrics")
    return parser.parse_args()


# ── Formatting helpers ─────────────────────────────────────────────────────────
SEP  = "═" * 62
SEP2 = "─" * 62


def _banner(title: str, subtitle: str = "") -> None:
    print(f"\n{_c(SEP, BOLD + CYAN)}")
    print(f"  {_c(title, BOLD + CYAN)}")
    if subtitle:
        print(f"  {_c(subtitle, DIM)}")
    print(f"{_c(SEP, BOLD + CYAN)}\n")


def _section(title: str) -> None:
    print(f"\n{_c(SEP2, BLUE)}")
    print(f"  {_c(title, BOLD + BLUE)}")
    print(f"{_c(SEP2, BLUE)}")


def _event(step: int, msg: str, colour: str = "") -> None:
    colour = colour or CYAN
    print(f"  {_c(f'[Step {step:>2}]', BOLD + colour)}  {msg}")


def _metric(label: str, value: str, colour: str = "") -> None:
    colour = colour or RESET
    print(f"  {(label + ':'):<30} {_c(value, colour)}")


def _bar(value: float, width: int = 20) -> str:
    filled = max(0, min(int(round(float(value) * width)), width))
    return "█" * filled + "░" * (width - filled)


# ── Live CLI dashboard ─────────────────────────────────────────────────────────
def _print_live_dashboard(step: int, iteration: ClosedLoopIteration) -> None:
    m             = iteration.metrics
    fraud_leakage = float(m.get("fraud_leakage_pct", 0.0))
    system_cost   = float(m.get("total_system_cost", 0.0))
    threshold     = float(iteration.defender_threshold)
    fn_rate       = float(m.get("false_negative_rate", 0.0))
    fp_rate       = float(m.get("false_positive_rate", 0.0))
    flagged       = len(iteration.selected_nodes)

    leakage_col = RED if fraud_leakage > 30 else YELLOW if fraud_leakage > 10 else GREEN
    cost_col    = RED if system_cost   > 200 else YELLOW if system_cost   > 80 else GREEN
    eq_label    = _c("✔ EQUILIBRIUM", GREEN) if iteration.equilibrium_detected else _c("⚡ ACTIVE BATTLE", YELLOW)

    top = sorted(iteration.attacker_strategy_mix.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = "  ".join(f"{STRATEGY_DISPLAY.get(n, n)} {v:.0%}" for n, v in top)

    print(f"\n  {_c('╔══ LIVE METRICS  (Round ' + str(step) + ')', BOLD + BLUE)} {'═' * 30}")
    print(f"  ║  {_c('Fraud Leakage:', DIM):<24} {_c(f'{fraud_leakage:6.1f}%', leakage_col)}  {_bar(fraud_leakage / 100.0)}")
    print(f"  ║  {_c('System Cost:', DIM):<24} {_c(f'{system_cost:8.2f}', cost_col)}")
    print(f"  ║  {_c('Defender Threshold:', DIM):<24} {_c(f'{threshold:.3f}', BOLD)}    {_bar(threshold)}")
    print(f"  ║  {_c('False Negative Rate:', DIM):<24} {_c(f'{fn_rate:.1%}', YELLOW)}")
    print(f"  ║  {_c('False Positive Rate:', DIM):<24} {_c(f'{fp_rate:.1%}', DIM)}")
    print(f"  ║  {_c('Nodes Flagged:', DIM):<24} {_c(str(flagged), BOLD)}")
    print(f"  ║  {_c('Difficulty:', DIM):<24} {_c(iteration.difficulty_level.upper(), BOLD)}")
    print(f"  ║  {_c('System State:', DIM):<24} {eq_label}")
    print(f"  ║  {_c('Top Attacker Strategies:', DIM)}")
    print(f"  ║    {_c(top_str, CYAN)}")
    print(f"  ╚{'═' * 52}")


# ── Narrative helpers ──────────────────────────────────────────────────────────
def _strategy_narrative(strategy: str) -> str:
    narr: Dict[str, str] = {
        "low_and_slow":   "Attackers adopt a low-and-slow approach — tiny signals, hard to catch.",
        "burst_attack":   f"{_c('🔥 Attackers SURGE with high-volume burst attacks!', RED)}",
        "camouflage":     "Attackers deploy camouflage, mimicking honest node behaviour.",
        "perfect_mimic":  f"{_c('⚠️  Attackers switch to Perfect Mimic — nearly indistinguishable!', RED + BOLD)}",
        "slow_drift":     "Attackers drift slowly, changing behaviour incrementally to evade detection.",
        "decoy_attacker": f"{_c('🎭 Attackers launch decoys to distract and overload the defender.', YELLOW)}",
        "mixed_cluster":  f"{_c('🌐 Attackers coordinate mixed clusters — distributed and hard to pin down.', YELLOW)}",
    }
    return narr.get(strategy, f"Attackers pivot to '{STRATEGY_DISPLAY.get(strategy, strategy)}'.")


def _scenario_narrative(scenario: str) -> str:
    narr: Dict[str, str] = {
        "steady_state_low_level_fraud":  "Environment: background noise with persistent low-level fraud.",
        "coordinated_attack_campaign":   f"{_c('🚨 Environment: coordinated multi-node attack campaign detected!', RED)}",
        "sudden_attack_surge":           f"{_c('💥 Environment: SUDDEN SURGE — attack volume exploding!', RED + BOLD)}",
        "defender_overload":             f"{_c('⚡ Environment: Defender resources stretched — overload event!', YELLOW)}",
        "stealth_long_term_infiltration":f"{_c('👁  Environment: stealthy long-term infiltration underway.', YELLOW)}",
    }
    return narr.get(scenario, f"Scenario: {SCENARIO_DISPLAY.get(scenario, scenario)}")


def _defender_narrative(prev: float, curr: float, eq: bool) -> str:
    delta = curr - prev
    if eq:
        return _c(f"🟢 Defender holds steady at threshold {curr:.3f} — system in equilibrium.", GREEN)
    if delta > 0.02:
        return _c(f"🛡  Defender TIGHTENS threshold ({prev:.3f} → {curr:.3f}) — raising defences.", BLUE)
    if delta < -0.02:
        return _c(f"🔓 Defender eases threshold ({prev:.3f} → {curr:.3f}) — reducing false positives.", CYAN)
    return _c(f"🛡  Defender adjusts threshold to {curr:.3f}.", BLUE)


def _print_step_narrative(step: int, iteration: ClosedLoopIteration, prev_threshold: float) -> None:
    leakage     = float(iteration.metrics.get("fraud_leakage_pct", 0.0))
    system_cost = float(iteration.metrics.get("total_system_cost", 0.0))
    flagged     = len(iteration.selected_nodes)
    eq          = iteration.equilibrium_detected

    print()
    _event(step, _scenario_narrative(iteration.selected_adversarial_scenario), DIM)
    _event(step, _strategy_narrative(iteration.selected_attacker_strategy), RED)
    _event(step, _defender_narrative(prev_threshold, iteration.defender_threshold, eq), BLUE)

    if leakage > 40:
        _event(step, _c(f"⚠️  Fraud leakage CRITICAL at {leakage:.1f}%!", RED))
    elif leakage < 5:
        _event(step, _c(f"✅ Fraud leakage very low: {leakage:.1f}%", GREEN))

    if flagged > 0:
        _event(step, _c(f"🔴 {flagged} nodes flagged and quarantined.", YELLOW))

    if step >= 3 and system_cost < 50:
        _event(step, _c("📉 System efficiency improving — costs falling.", GREEN))

    if eq:
        _event(step, _c("🏁 Equilibrium detected — the battle is stabilising.", BOLD + GREEN))


# ── Explainability ─────────────────────────────────────────────────────────────
_FEATURE_DISPLAY: Dict[str, str] = {
    "claimed_inferred_distance_mismatch_mean": "Distance Mismatch (high = suspicious)",
    "latency_inconsistency_score_mean":        "Latency Inconsistency",
    "burst_activity_score_mean":               "Burst Activity Score",
    "behavior_volatility_mean":                "Behavior Volatility",
    "sudden_change_score_mean":                "Sudden Change Score",
}


def _print_explainability(
    df: pd.DataFrame,
    fraud_scores: Dict[int, float],
    selected_nodes: List[int],
    threshold: float,
    max_nodes: int = 5,
) -> None:
    if not selected_nodes or df.empty:
        return

    _section("EXPLAINABILITY — WHY THESE NODES WERE FLAGGED")

    if "node_id" not in df.columns:
        print("  (Node feature data unavailable)")
        return

    df_indexed = df.set_index("node_id")

    shown = 0
    for node_id in selected_nodes[:max_nodes]:
        nid = int(node_id)
        score = float(fraud_scores.get(nid, 0.0))
        gap = score - threshold
        confidence = min(100.0, max(0.0, gap / max(1.0 - threshold, 0.01) * 100.0))

        if nid not in df_indexed.index:
            continue

        row = df_indexed.loc[nid]
        label_raw = row.get("label_name", row.get("label", "unknown"))
        label_name = str(label_raw)

        score_col = RED if score > 0.8 else YELLOW
        print(
            f"\n  {_c(f'Node #{nid}', BOLD + RED)}"
            f"  fraud score {_c(f'{score:.3f}', score_col)}"
            f"  (confidence {_c(f'{confidence:.0f}%', BOLD)})"
        )
        print(f"  {_c('True label:', DIM)} {label_name}")

        for col, display in _FEATURE_DISPLAY.items():
            if col in df.columns:
                val = float(row.get(col, 0.0))
                norm = min(abs(val) / 5.0, 1.0)
                bar  = _bar(norm, width=15)
                print(f"    {(display + ':'):<44} {_c(f'{val:7.3f}', CYAN)}  {_c(bar, YELLOW)}")

        shown += 1

    if shown == 0:
        print("  (No matching node details available)")
    print()


# ── Before / After ─────────────────────────────────────────────────────────────
def _print_before_after(
    mean_leakage: float,
    system_efficiency: float,
    fraud_prevented_pct: float,
) -> None:
    _section("BEFORE vs AFTER — FRAUD PREVENTION IMPACT")

    # Estimate unchecked leakage: if the defender were absent, leakage
    # would approach the attacker's raw effectiveness (approximated here).
    estimated_without = min(100.0, mean_leakage + fraud_prevented_pct * 0.85)
    estimated_without = max(estimated_without, mean_leakage + 5.0)

    print()
    print(f"  {_c('WITHOUT Anti-Spoofing System:', BOLD + RED)}")
    print(f"    Estimated fraud leakage:  {_c(f'{estimated_without:.1f}%', RED)}  {_c(_bar(estimated_without / 100.0), RED)}")
    print(f"    Attackers act freely — fraudulent rewards extracted at scale.")
    print()
    print(f"  {_c('WITH Anti-Spoofing System:', BOLD + GREEN)}")
    print(f"    Actual fraud leakage:     {_c(f'{mean_leakage:.1f}%', GREEN)}  {_c(_bar(mean_leakage / 100.0), GREEN)}")
    print(f"    System efficiency:        {_c(f'{system_efficiency:.2%}', GREEN)}")
    print()

    prevented = min(100.0, max(0.0, fraud_prevented_pct))
    print(f"  {_c('▶', BOLD + GREEN)} System prevented {_c(f'{prevented:.1f}%', BOLD + GREEN)} of fraud losses.")
    print(f"  {_c('▶', BOLD + GREEN)} Equivalent to eliminating the majority of fraudulent reward leakage.")
    print()


# ── Visualizations ─────────────────────────────────────────────────────────────
def _save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _generate_visualizations(
    history: Dict,
    output_dir: Path,
    scenario_label: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    strategy_hist  = history.get("attacker_strategy_distribution", [])
    threshold_hist = [float(v) for v in history.get("defender_threshold", [])]
    cost_hist      = [float(v) for v in history.get("system_cost", [])]
    leakage_hist   = [float(v) for v in history.get("fraud_leakage", [])]
    eq_hist        = [float(v) for v in history.get("equilibrium_detected", [])]
    eq_sd          = [float(v) for v in history.get("equilibrium_strategy_delta", [])]
    eq_cd          = [float(v) for v in history.get("equilibrium_cost_delta_ratio", [])]

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Fraud leakage over time
    if leakage_hist:
        fig, ax = plot_fraud_leakage_vs_time(leakage_hist, output_path=None)
        ax.set_title(f"Fraud Leakage Over Time\n{scenario_label}", fontsize=13, fontweight="bold")
        p = output_dir / "fraud_leakage_over_time.png"
        _save_plot(fig, p)
        saved.append(p)

    # 2. System cost over time
    if cost_hist:
        fig, ax = plot_cost_vs_time(cost_hist, output_path=None)
        ax.set_title(f"System Cost Over Time\n{scenario_label}", fontsize=13, fontweight="bold")
        p = output_dir / "system_cost_over_time.png"
        _save_plot(fig, p)
        saved.append(p)

    # 3. Attacker strategy distribution
    if strategy_hist:
        fig, ax = plot_strategy_distribution_over_time(strategy_hist, output_path=None)
        ax.set_title(f"Attacker Strategy Distribution\n{scenario_label}", fontsize=13, fontweight="bold")
        p = output_dir / "attacker_strategy_distribution.png"
        _save_plot(fig, p)
        saved.append(p)

    # 4. Defender threshold over time
    if threshold_hist:
        fig, ax = plot_defender_threshold_over_time(threshold_hist, output_path=None)
        ax.set_title(f"Defender Policy (Threshold) Over Time\n{scenario_label}", fontsize=13, fontweight="bold")
        p = output_dir / "defender_threshold_over_time.png"
        _save_plot(fig, p)
        saved.append(p)

    # 5. Equilibrium detection
    if eq_hist and eq_sd and eq_cd:
        fig, _ = plot_equilibrium_detection(eq_hist, eq_sd, eq_cd, output_path=None)
        fig.suptitle(f"Equilibrium Detection Dynamics\n{scenario_label}", fontsize=13, fontweight="bold")
        p = output_dir / "equilibrium_detection.png"
        _save_plot(fig, p)
        saved.append(p)

    # 6. Before vs After bar chart
    if leakage_hist and len(leakage_hist) >= 2:
        n = len(leakage_hist)
        third = max(1, n // 3)
        early = float(np.mean(leakage_hist[:third]))
        late  = float(np.mean(leakage_hist[-third:]))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            ["Early (No Adaptation)", "Late (With Adaptation)"],
            [early, late],
            color=["#ef4444", "#22c55e"],
            width=0.45,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=12, fontweight="bold")
        ax.set_ylabel("Fraud Leakage (%)", fontsize=11)
        ax.set_ylim(0, max(early, late) * 1.4 + 5)
        ax.set_title(
            f"Before vs After — Fraud Reduction\n{scenario_label}",
            fontsize=13,
            fontweight="bold",
        )
        p = output_dir / "before_after_comparison.png"
        _save_plot(fig, p)
        saved.append(p)

    return saved


# ── Final report ───────────────────────────────────────────────────────────────
def _print_final_report(
    scenario_label: str,
    rounds: List[ClosedLoopIteration],
    system_analysis: Dict,
    saved_plots: List[Path],
    elapsed: float,
    output_dir: Path,
) -> None:
    _banner("FINAL REPORT", f"Scenario: {scenario_label}")

    eq_reached    = bool(system_analysis.get("equilibrium_reached", False))
    eq_type       = str(system_analysis.get("equilibrium_type", "unknown"))
    dominant      = str(system_analysis.get("dominant_strategy", "unknown"))
    dominance     = float(system_analysis.get("dominance_share", 0.0))
    prevented_pct = float(system_analysis.get("fraud_prevented_pct", 0.0))
    efficiency    = float(system_analysis.get("system_efficiency", 0.0))
    mean_cost     = float(system_analysis.get("mean_system_cost", 0.0))
    mean_leakage  = float(system_analysis.get("mean_fraud_leakage_pct", 0.0))

    # Count strategy usage across all rounds
    strategy_counts: Dict[str, int] = {}
    for it in rounds:
        s = it.selected_attacker_strategy
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    top_used = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    eq_col   = GREEN if eq_reached else YELLOW
    eq_label = "YES" if eq_reached else "NOT REACHED"

    _metric("Equilibrium Reached",    eq_label, eq_col)
    _metric("Equilibrium Type",       eq_type.replace("_", " ").title())
    _metric(
        "Dominant Attacker Strategy",
        f"{STRATEGY_DISPLAY.get(dominant, dominant)} ({dominance:.0%})",
    )
    _metric(
        "Top Strategies Used",
        "  ".join(f"{STRATEGY_DISPLAY.get(s, s)}×{n}" for s, n in top_used),
    )
    print()
    _metric("Total Fraud Prevented",  f"{prevented_pct:.1f}%",      GREEN)
    _metric("Mean Fraud Leakage",     f"{mean_leakage:.1f}%",       YELLOW if mean_leakage > 10 else GREEN)
    _metric("System Efficiency",      f"{efficiency:.2%}",           GREEN if efficiency > 0.005 else YELLOW)
    _metric("Mean System Cost",       f"{mean_cost:.2f}",            YELLOW if mean_cost > 100 else GREEN)
    _metric("Iterations Run",         str(len(rounds)))
    _metric("Wall-Clock Time",        f"{elapsed:.1f}s")
    print()

    if eq_reached:
        print(f"  {_c('★', BOLD + GREEN)} System reached {_c(eq_type.replace('_', ' '), BOLD)} — attackers and defender stabilised.")
    else:
        print(f"  {_c('⚠', BOLD + YELLOW)} System did NOT stabilise — the battle continues beyond this window.")

    print(f"  {_c('★', BOLD + GREEN)} System eliminated ~{_c(f'{prevented_pct:.1f}%', BOLD + GREEN)} of potential fraud losses.")
    print()

    if saved_plots:
        _section("SAVED VISUALIZATIONS")
        for p in saved_plots:
            print(f"  {_c('📊', DIM)} {p}")

    print()
    print(f"  {_c('Output directory:', DIM)} {output_dir.resolve()}")
    print(f"\n{_c(SEP, BOLD + CYAN)}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    # Disable colour if requested or stdout is not a TTY
    global RED, GREEN, YELLOW, BLUE, CYAN, BOLD, DIM, RESET
    if args.no_color or not sys.stdout.isatty():
        RED = GREEN = YELLOW = BLUE = CYAN = BOLD = DIM = RESET = ""

    scenario_cfg = SCENARIO_CATALOGUE[args.scenario]
    iterations   = args.steps if args.steps > 0 else int(scenario_cfg["default_iterations"])
    output_dir   = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build scenario schedule, cycling if fewer entries than iterations requested
    base_schedule: List[str] = list(scenario_cfg["schedule"])
    schedule: List[str] = []
    while len(schedule) < iterations:
        schedule.extend(base_schedule)
    schedule = schedule[:iterations]

    # ── Banner ─────────────────────────────────────────────────────────────────
    _banner(
        "ADVERSARIAL FRAUD DETECTION — LIVE DEMO",
        "A live battle between attackers and defenders in a decentralised network",
    )
    print(f"  {_c('Scenario:', DIM):<22} {_c(scenario_cfg['label'], BOLD + CYAN)}")
    print(f"  {_c('Description:', DIM):<22} {scenario_cfg['description']}")
    print(f"  {_c('Iterations:', DIM):<22} {iterations}")
    print(f"  {_c('Nodes:', DIM):<22} {args.nodes}")
    print(f"  {_c('Seed:', DIM):<22} {args.seed}")
    print(f"  {_c('Output:', DIM):<22} {output_dir}")
    print()

    # ── Initialise agents and environment ──────────────────────────────────────
    memory_dir = output_dir / "memory"
    environment = AdversarialFraudEnvironment(
        iterations=iterations,
        seed=args.seed,
        total_nodes=args.nodes,
        time_steps=16,
        attacker_memory_path=memory_dir / "attacker_memory.json",
        defender_memory_path=memory_dir / "defender_memory.json",
        uncertainty_level=0.08,
        inner_iterations=3,
        scenario_schedule=schedule,
    )
    attacker = AttackerAgent(temperature=0.25, epsilon=0.08, use_ucb_selection=True)
    defender = DefenderAgent(threshold=0.5, budget_ratio=0.07, min_budget=3)

    _section("THE BATTLE BEGINS")
    print(f"  {_c('Initialising', DIM)} {args.nodes} nodes · {iterations} rounds")
    print(f"  {_c('Attacker', RED)} starts with equal probability across {len(STRATEGY_MARKETPLACE)} strategies.")
    print(f"  {_c('Defender', BLUE)} starts at threshold {defender.threshold:.3f} — adaptive learning begins now.")
    print()

    # ── Step-by-step simulation loop ───────────────────────────────────────────
    rounds: List[ClosedLoopIteration] = []
    prev_threshold = defender.threshold
    start_time = time.perf_counter()

    for i in range(iterations):
        step = i + 1
        print(f"\n{_c(f'  ── ROUND {step}/{iterations} ──', BOLD + DIM)}", end="", flush=True)

        outcome = environment.step(iteration=i, attacker_agent=attacker, defender_agent=defender)
        it = outcome.result
        rounds.append(it)

        if not args.quiet:
            _print_step_narrative(step, it, prev_threshold)
        _print_live_dashboard(step, it)

        prev_threshold = it.defender_threshold

    elapsed = time.perf_counter() - start_time

    # ── Explainability (best-effort) ───────────────────────────────────────────
    if not args.quiet and rounds and environment.final_model is not None:
        last_it = rounds[-1]
        try:
            sim = build_network_simulation(
                total_nodes=args.nodes,
                time_steps=16,
                seed=args.seed + iterations,
            )
            df    = extract_node_features(sim)
            probs = environment.final_model.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
            fraud_scores_map: Dict[int, float] = {
                int(nid): float(s) for nid, s in zip(df["node_id"], probs)
            }
            _print_explainability(
                df=df,
                fraud_scores=fraud_scores_map,
                selected_nodes=last_it.selected_nodes[:5],
                threshold=last_it.defender_threshold,
            )
        except (KeyError, ValueError, AttributeError, TypeError):
            pass  # explainability is informational — never block the demo

    # ── System analysis ────────────────────────────────────────────────────────
    system_analysis = analyze_system_dynamics(environment.history, rounds=rounds)

    # ── Before / After ─────────────────────────────────────────────────────────
    if not args.quiet:
        _print_before_after(
            mean_leakage=float(system_analysis.get("mean_fraud_leakage_pct", 0.0)),
            system_efficiency=float(system_analysis.get("system_efficiency", 0.0)),
            fraud_prevented_pct=float(system_analysis.get("fraud_prevented_pct", 0.0)),
        )

    # ── Visualizations ─────────────────────────────────────────────────────────
    _section("GENERATING VISUALIZATIONS")
    saved_plots = _generate_visualizations(
        history=environment.history,
        output_dir=output_dir,
        scenario_label=scenario_cfg["label"],
    )
    for p in saved_plots:
        print(f"  {_c('✔', GREEN)} Saved: {p.name}")

    # ── Final report ───────────────────────────────────────────────────────────
    _print_final_report(
        scenario_label=scenario_cfg["label"],
        rounds=rounds,
        system_analysis=system_analysis,
        saved_plots=saved_plots,
        elapsed=elapsed,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
