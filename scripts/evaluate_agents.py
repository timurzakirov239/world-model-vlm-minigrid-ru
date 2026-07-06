from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs import make_minigrid_env, render_rgb, state_from_env  # noqa: E402
from src.planner import PlannerConfig, RandomShootingPlanner, load_vlm_scorer, load_world_model  # noqa: E402
from src.scorer import GoalDistanceScorer  # noqa: E402


def evaluate_mode(
    mode: str,
    args: argparse.Namespace,
    planner: RandomShootingPlanner | None,
    seeds: list[int],
) -> list[dict[str, float | int | str]]:
    results = []
    gif_dir = args.out_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        env = make_minigrid_env(args.env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        frames = [render_rgb(env)]
        total_reward = 0.0
        terminated = False
        truncated = False
        planner_info = []

        rng = np.random.default_rng(seed + 10_000)
        for step in range(args.max_steps):
            state = state_from_env(env)
            if mode == "random":
                action = int(rng.integers(3))
                info = {"objective": 0.0}
            else:
                if planner is None:
                    raise RuntimeError("Planner is required for world-model modes.")
                action, info = planner.plan(state, mode=mode)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            frames.append(render_rgb(env))
            planner_info.append(info)
            if terminated or truncated:
                break

        env.close()
        success = bool(total_reward > 0.0 or terminated)
        result = {
            "mode": mode,
            "seed": seed,
            "return": float(total_reward),
            "success": int(success),
            "steps": len(frames) - 1,
            "terminated": int(terminated),
            "truncated": int(truncated),
        }
        results.append(result)

        if len([r for r in results if r["mode"] == mode]) <= args.gif_episodes:
            safe_env = args.env_id.replace("/", "_").replace(":", "_")
            imageio.mimsave(
                gif_dir / f"{safe_env}_{mode}_seed_{seed}.gif",
                frames,
                duration=0.22,
            )

        if planner_info:
            info_path = args.out_dir / f"{mode}_seed_{seed}_planner_info.json"
            info_path.write_text(json.dumps(planner_info, indent=2), encoding="utf-8")

    return results


def summarize(results: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    modes = sorted({str(row["mode"]) for row in results})
    rows = []
    for mode in modes:
        subset = [row for row in results if row["mode"] == mode]
        returns = np.asarray([float(row["return"]) for row in subset], dtype=np.float32)
        successes = np.asarray([int(row["success"]) for row in subset], dtype=np.float32)
        steps = np.asarray([int(row["steps"]) for row in subset], dtype=np.float32)
        rows.append(
            {
                "mode": mode,
                "episodes": len(subset),
                "success_rate": float(successes.mean()),
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std()),
                "mean_steps": float(steps.mean()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(seeds: str) -> list[int]:
    return [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baselines and world-model planners.")
    parser.add_argument("--env-id", default="MiniGrid-Empty-5x5-v0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "empty5_rssm" / "rssm_world_model.pt",
    )
    parser.add_argument(
        "--vlm-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "vlm_cache" / "empty5_clip_goal.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eval" / "empty5",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=384)
    parser.add_argument("--vlm-weight", type=float, default=0.8)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.1)
    parser.add_argument("--gif-episodes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    model = load_world_model(args.checkpoint)
    vlm_scorer = load_vlm_scorer(args.vlm_cache)
    distance_scorer = GoalDistanceScorer(width=5, height=5)
    config = PlannerConfig(
        horizon=args.horizon,
        candidates=args.candidates,
        vlm_weight=args.vlm_weight,
        reward_weight=args.reward_weight,
        distance_weight=args.distance_weight,
    )
    planner = RandomShootingPlanner(
        model=model,
        config=config,
        scorer=vlm_scorer,
        distance_scorer=distance_scorer,
        seed=123,
    )

    all_results = []
    for mode in ["random", "wm_reward", "wm_vlm"]:
        print(f"Evaluating {mode}...")
        all_results.extend(evaluate_mode(mode, args, planner if mode != "random" else None, seeds))

    summary = summarize(all_results)
    write_csv(args.out_dir / "episodes.csv", all_results)
    write_csv(args.out_dir / "summary.csv", summary)
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "env_id": args.env_id,
                "seeds": seeds,
                "max_steps": args.max_steps,
                "horizon": args.horizon,
                "candidates": args.candidates,
                "vlm_weight": args.vlm_weight,
                "reward_weight": args.reward_weight,
                "distance_weight": args.distance_weight,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Summary:")
    for row in summary:
        print(
            f"{row['mode']}: success={row['success_rate']:.3f}, "
            f"return={row['mean_return']:.3f}±{row['std_return']:.3f}, "
            f"steps={row['mean_steps']:.1f}"
        )
    print(f"Saved evaluation to: {args.out_dir}")


if __name__ == "__main__":
    main()
