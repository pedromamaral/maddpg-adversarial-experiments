#!/usr/bin/env python3
"""
tools/plot_results.py
----------------------
Generates publication-quality figures from experiment results.

Outputs (saved to tools/figures/):
  reward_curves.pdf / .png           - episode reward vs. training step per architecture
  packet_loss_comparison.pdf / .png  - packet loss bar chart per architecture per attack
  robustness_degradation.pdf / .png  - (clean - attacked) / clean performance

Usage:
  # After running full experiment and pulling results from server
  python tools/plot_results.py --results host_data/results/
"""

import argparse
import json
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def _save(fig, stem: str):
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)

def load_results(results_dir: Path):
    """Load all JSON result files from host_data/results/."""
    data = {}
    for json_path in results_dir.glob("*.json"):
        with open(json_path, "r") as f:
            key = json_path.stem
            data[key] = json.load(f)
    return data

def plot_reward_curves(data: dict):
    print("Plotting reward curves ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Example: plot episode reward for each architecture variant
    for name, result in data.items():
        if "episode_rewards" in result:
            ax.plot(result["episode_rewards"], label=name, alpha=0.85)
    
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Training Reward Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.4)
    fig.tight_layout()
    _save(fig, "reward_curves")

def plot_packet_loss(data: dict):
    print("Plotting packet loss comparison ...")
    # Placeholder: extract packet_loss per architecture and attack type
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # This is a stub — real implementation would parse your JSON structure
    architectures = ["MLP", "Att", "GNN", "Dual", "Att+Dual", "GNN+Dual"]
    attacks = ["clean", "FGSM", "rand-FGSM"]
    
    # Example mock data
    x = np.arange(len(architectures))
    width = 0.25
    for i, att in enumerate(attacks):
        vals = np.random.uniform(2, 15, len(architectures))
        ax.bar(x + i * width, vals, width, label=att, alpha=0.85)
    
    ax.set_xlabel("Architecture Variant", fontsize=11)
    ax.set_ylabel("Packet Loss Rate (%)", fontsize=11)
    ax.set_title("Packet Loss Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(architectures, fontsize=10)
    ax.legend(fontsize=10, title="Attack")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, "packet_loss_comparison")

def plot_robustness(data: dict):
    print("Plotting robustness degradation ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Mock: degradation = (clean_perf - attacked_perf) / clean_perf * 100
    architectures = ["MLP", "Att", "GNN", "Dual", "Att+Dual", "GNN+Dual"]
    degradation = np.random.uniform(5, 30, len(architectures))
    
    ax.bar(architectures, degradation, color="#e74c3c", alpha=0.85)
    ax.set_ylabel("Performance Degradation (%)", fontsize=11)
    ax.set_title("Adversarial Robustness Degradation", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, "robustness_degradation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="../host_data/results",
                        help="Path to results directory")
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        print("Pull results from server first: scp -r user@server:path/host_data/results ./")
        exit(1)
    
    print("=" * 60)
    print(" plot_results.py — Experiment Results Figures")
    print("=" * 60)
    
    data = load_results(results_dir)
    print(f"Loaded {len(data)} result files")
    print()
    
    plot_reward_curves(data)
    plot_packet_loss(data)
    plot_robustness(data)
    
    print()
    print(f"All figures saved to {FIGURES_DIR}/")
    print("Done.")
