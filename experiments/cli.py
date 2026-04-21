from __future__ import annotations

import argparse

from core_engine.loop import run_closed_loop
from features import extract_node_features
from modeling import benchmark_fraud_models, train_fraud_model
from core_engine.orchestrator import run_pipeline
from simulation import build_network_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop adversarial intelligence CLI")
    parser.add_argument("command", choices=["run_simulation", "train_model", "run_closed_loop", "benchmark"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=220)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    if args.command == "run_simulation":
        sim = build_network_simulation(total_nodes=args.nodes, time_steps=args.steps, seed=args.seed, difficulty_level="hard")
        print({"nodes": len(sim.nodes), "time_steps": len(sim.time_steps), "scenario_metrics": sim.scenario_metrics})
        return

    if args.command == "train_model":
        sim = build_network_simulation(total_nodes=args.nodes, time_steps=args.steps, seed=args.seed)
        artifacts = train_fraud_model(extract_node_features(sim), random_state=args.seed)
        print({"selected_model": artifacts.selected_model_name, "metrics": artifacts.metrics})
        return

    if args.command == "run_closed_loop":
        result = run_closed_loop(iterations=args.rounds, seed=args.seed, total_nodes=args.nodes, time_steps=args.steps)
        print({
            "rounds": [r.__dict__ for r in result.iterations],
            "final_metrics": result.final_model.metrics,
            "benchmark": result.benchmark.to_dict(orient="records"),
        })
        return

    if args.command == "benchmark":
        artifacts = run_pipeline(seed=args.seed)
        print(artifacts.benchmark.to_dict(orient="records"))
        return

    raise ValueError("unsupported command")


if __name__ == "__main__":
    main()
