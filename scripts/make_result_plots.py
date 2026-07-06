from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(eval_dir: Path) -> None:
    summary = pd.read_csv(eval_dir / "summary.csv")
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "random": "Random",
        "wm_reward": "WM planning",
        "wm_vlm": "WM + VLM",
    }
    colors = {
        "random": "#7c7c7c",
        "wm_reward": "#3b82f6",
        "wm_vlm": "#16a34a",
    }
    order = ["random", "wm_reward", "wm_vlm"]
    summary = summary.set_index("mode").loc[order].reset_index()

    plt.figure(figsize=(6.2, 4.0))
    plt.bar(
        [label_map[mode] for mode in summary["mode"]],
        summary["success_rate"],
        color=[colors[mode] for mode in summary["mode"]],
    )
    plt.ylim(0, 1)
    plt.ylabel("Success rate")
    plt.title("MiniGrid-Empty-5x5-v0 evaluation")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 4.0))
    plt.bar(
        [label_map[mode] for mode in summary["mode"]],
        summary["mean_return"],
        yerr=summary["std_return"],
        color=[colors[mode] for mode in summary["mode"]],
        capsize=4,
    )
    plt.ylabel("Mean return")
    plt.title("Average episodic return")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_return.png", dpi=180)
    plt.close()

    print(f"Saved plots to: {plots_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create result plots for the report.")
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/eval/empty5_tuned"))
    return parser.parse_args()


if __name__ == "__main__":
    make_plots(parse_args().eval_dir)
