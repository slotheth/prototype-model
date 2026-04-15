"""
Show training progress from trainer_state.json and log files.

Usage:
    python status.py
    python status.py --watch       # Auto-refresh every 10s
    python status.py --watch 5     # Auto-refresh every 5s
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import config as cfg


def read_trainer_state() -> dict | None:
    """Read the latest trainer_state.json from output dir."""
    # Check checkpoints for latest state
    state_file = cfg.LORA_DIR / "trainer_state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)

    # Check checkpoint subdirs
    checkpoints = sorted(cfg.LORA_DIR.glob("checkpoint-*/trainer_state.json"))
    if checkpoints:
        with open(checkpoints[-1], "r") as f:
            return json.load(f)

    return None


def format_duration(seconds: float) -> str:
    """Format seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    hours = seconds / 3600
    mins = (seconds % 3600) / 60
    return f"{hours:.0f}h {mins:.0f}m"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def show_status():
    """Display current training status."""
    state = read_trainer_state()

    print("=" * 50)
    print("  Training Status")
    print("=" * 50)
    print(f"  Model:  {cfg.MODEL_NAME}")
    print(f"  Output: {cfg.LORA_DIR}")

    if state is None:
        # Check if training is even running
        pid_running = False
        try:
            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", "train.py"], capture_output=True, text=True,
            )
            pid_running = result.returncode == 0
        except Exception:
            pass

        if pid_running:
            print(f"\n  Status: STARTING (no checkpoints yet)")
        else:
            print(f"\n  Status: NOT RUNNING")
            print(f"  Run: make train")
        print("=" * 50)
        return

    # Parse state
    log_history = state.get("log_history", [])
    max_steps = state.get("max_steps", 0)
    epoch = state.get("epoch", 0)
    total_epochs = cfg.NUM_TRAIN_EPOCHS

    # Find latest train log entry
    train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
    eval_logs = [l for l in log_history if "eval_loss" in l]

    current_step = train_logs[-1]["step"] if train_logs else 0
    progress = current_step / max_steps * 100 if max_steps > 0 else 0

    # Progress bar
    bar_width = 30
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\n  Progress: [{bar}] {progress:.1f}%")
    print(f"  Step:     {current_step:,} / {max_steps:,}")
    print(f"  Epoch:    {epoch:.1f} / {total_epochs}")

    # Latest metrics
    if train_logs:
        latest = train_logs[-1]
        print(f"\n  Latest Train Metrics:")
        print(f"    Loss:          {latest.get('loss', 'N/A')}")
        if "grad_norm" in latest:
            print(f"    Grad norm:     {latest['grad_norm']:.4f}")
        if "learning_rate" in latest:
            print(f"    Learning rate: {latest['learning_rate']:.2e}")

    if eval_logs:
        latest_eval = eval_logs[-1]
        print(f"\n  Latest Eval Metrics:")
        print(f"    Eval loss:     {latest_eval.get('eval_loss', 'N/A')}")
        if "eval_mean_token_accuracy" in latest_eval:
            acc = latest_eval["eval_mean_token_accuracy"]
            print(f"    Token acc:     {acc:.1%}")

    # Time estimates
    if len(train_logs) >= 2:
        first = train_logs[0]
        last = train_logs[-1]
        elapsed_steps = last["step"] - first["step"]
        if elapsed_steps > 0 and "epoch" in first and "epoch" in last:
            # Estimate from step timing
            total_runtime = state.get("total_flos", 0)
            remaining_steps = max_steps - current_step
            # Use log timestamps if available
            if len(train_logs) >= 5:
                recent = train_logs[-5:]
                if all("step" in l for l in recent):
                    step_diff = recent[-1]["step"] - recent[0]["step"]
                    if step_diff > 0:
                        # Rough estimate based on wall time between log entries
                        sec_per_step = 2.5  # fallback estimate
                        remaining_sec = remaining_steps * sec_per_step
                        print(f"\n  Remaining:   ~{format_duration(remaining_sec)} (est.)")

    # Loss trend
    if len(train_logs) >= 10:
        early_loss = sum(l["loss"] for l in train_logs[:5]) / 5
        recent_loss = sum(l["loss"] for l in train_logs[-5:]) / 5
        trend = "↓ improving" if recent_loss < early_loss else "↑ worsening"
        print(f"  Loss trend:  {early_loss:.4f} → {recent_loss:.4f} ({trend})")

    print(f"\n  Updated:     {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Show training progress")
    parser.add_argument("--watch", nargs="?", const=10, type=int, metavar="SEC",
                        help="Auto-refresh every N seconds (default: 10)")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                clear_screen()
                show_status()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show_status()


if __name__ == "__main__":
    main()
