"""
plot_learning_curve.py — Visualize GRPO training progress from HF Trainer checkpoints.

The HF Trainer automatically saves `trainer_state.json` inside every checkpoint folder.
This script reads all those files, extracts per-step metrics, and plots a learning curve.

Usage (run in Colab after training):
    python evaluation/plot_learning_curve.py \\
        --checkpoint_dir "/content/drive/MyDrive/indicators_grpo_v2"

Output:
    - learning_curve.png saved to the checkpoint_dir
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # Headless (no display needed in Colab)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _load_trainer_state(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    Scan all checkpoint-* subdirectories and collect log history
    from trainer_state.json.
    """
    base = Path(checkpoint_dir)
    all_logs = []
    seen_steps = set()

    checkpoints = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )

    for ckpt in checkpoints:
        state_file = ckpt / "trainer_state.json"
        if not state_file.exists():
            print(f"  ⚠️  No trainer_state.json in {ckpt.name}")
            continue

        with open(state_file) as f:
            state = json.load(f)

        for entry in state.get("log_history", []):
            step = entry.get("step")
            if step is not None and step not in seen_steps:
                seen_steps.add(step)
                all_logs.append(entry)

    all_logs.sort(key=lambda x: x.get("step", 0))
    print(f"  ✅ Loaded {len(all_logs)} log entries from {len(checkpoints)} checkpoints")
    return all_logs


def _extract_series(logs: List[Dict], key: str):
    """Extract (steps, values) for a given metric key."""
    steps, values = [], []
    for entry in logs:
        if key in entry and "step" in entry:
            steps.append(entry["step"])
            values.append(entry[key])
    return steps, values


def plot_learning_curve(checkpoint_dir: str) -> None:
    print(f"[Plot] Reading trainer logs from: {checkpoint_dir}")
    logs = _load_trainer_state(checkpoint_dir)

    if not logs:
        print("❌ No log entries found. Make sure trainer_state.json exists in checkpoint folders.")
        return

    # Extract all available metrics
    reward_steps, reward_vals = _extract_series(logs, "reward")
    loss_steps, loss_vals = _extract_series(logs, "loss")
    train_loss_steps, train_loss_vals = _extract_series(logs, "train_loss")

    # Fallback: use train_loss if loss not found
    if not loss_vals and train_loss_vals:
        loss_steps, loss_vals = train_loss_steps, train_loss_vals

    # ── Build Plot ──────────────────────────────────────────────────────────
    n_plots = sum([bool(reward_vals), bool(loss_vals)])
    if n_plots == 0:
        print("❌ No 'reward' or 'loss' keys found in log history.")
        print("   Available keys:", list(set(k for e in logs for k in e.keys())))
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(
        "IndicatorsEnv — GRPO Training Learning Curve\n"
        "Qwen2.5-1.5B-Instruct + QLoRA | Anti-Bias Reward",
        fontsize=13, fontweight="bold"
    )

    ax_idx = 0

    if reward_vals:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(reward_steps, reward_vals, color="#4CAF50", linewidth=2, alpha=0.8, label="Per-step Reward")
        # Rolling average
        if len(reward_vals) >= 10:
            import statistics
            window = 10
            rolled = [
                statistics.mean(reward_vals[max(0, i-window//2):i+window//2+1])
                for i in range(len(reward_vals))
            ]
            ax.plot(reward_steps, rolled, color="#1B5E20", linewidth=2.5, linestyle="--", label="Rolling Avg (10)")
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Reward", fontsize=11)
        ax.set_title("Reward per Step", fontsize=12)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="red", linestyle=":", alpha=0.5, label="Break-even")

    if loss_vals:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(loss_steps, loss_vals, color="#2196F3", linewidth=2, alpha=0.8, label="Training Loss")
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Training Loss", fontsize=12)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(checkpoint_dir) / "learning_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Learning curve saved to: {out_path}")
    print("   Add this image to your mentor report as Section 4b.")

    # Also print raw numbers
    if reward_vals:
        mid = len(reward_vals) // 2
        print(f"\n  First 10 steps avg reward: {sum(reward_vals[:10])/10:.4f}")
        print(f"  Mid     10 steps avg reward: {sum(reward_vals[mid:mid+10])/10:.4f}")
        print(f"  Last    10 steps avg reward: {sum(reward_vals[-10:])/10:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GRPO training learning curve from HF checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        default="/content/drive/MyDrive/indicators_grpo_v2",
        help="Path to the output_dir used during training"
    )
    args = parser.parse_args()
    plot_learning_curve(args.checkpoint_dir)
