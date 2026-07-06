from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs import (  # noqa: E402
    get_raw_state,
    make_minigrid_env,
    normalize_state,
    render_rgb,
    set_raw_state,
    valid_empty_positions,
)


NAVIGATION_ACTIONS = [0, 1, 2]  # left, right, forward


def collect_transition_dataset(
    env_id: str,
    out_path: Path,
    seed: int,
    repeats: int,
    gif: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_minigrid_env(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    width, height = env.unwrapped.width, env.unwrapped.height

    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    raw_states = []
    raw_next_states = []
    episode_ids = []
    timesteps = []
    frames = []

    transition_idx = 0
    positions = valid_empty_positions(env)

    for repeat in range(repeats):
        for x, y in positions:
            for direction in range(4):
                for action in NAVIGATION_ACTIONS:
                    env.reset(seed=seed)
                    raw_state = np.asarray([x, y, direction, width - 2, height - 2], dtype=np.float32)
                    set_raw_state(env, raw_state, step_count=repeat)
                    if gif and len(frames) < 16:
                        frames.append(render_rgb(env))

                    _, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    raw_next_state = get_raw_state(env)

                    states.append(normalize_state(raw_state, width, height))
                    next_states.append(normalize_state(raw_next_state, width, height))
                    actions.append(action)
                    rewards.append(float(reward))
                    dones.append(done)
                    raw_states.append(raw_state)
                    raw_next_states.append(raw_next_state)
                    episode_ids.append(repeat)
                    timesteps.append(transition_idx)
                    transition_idx += 1

                    if gif and len(frames) < 16:
                        frames.append(render_rgb(env))

    env.close()

    np.savez_compressed(
        out_path,
        states=np.asarray(states, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        raw_states=np.asarray(raw_states, dtype=np.float32),
        raw_next_states=np.asarray(raw_next_states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        env_id=np.asarray(env_id),
        seed=np.asarray(seed, dtype=np.int64),
        action_ids=np.asarray(NAVIGATION_ACTIONS, dtype=np.int64),
        width=np.asarray(width, dtype=np.int64),
        height=np.asarray(height, dtype=np.int64),
    )

    if gif and frames:
        gif_dir = PROJECT_ROOT / "artifacts" / "dataset_gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        safe_env = env_id.replace("/", "_").replace(":", "_")
        imageio.mimsave(gif_dir / f"{safe_env}_transition_samples.gif", frames, duration=0.3)

    print(f"Saved transition dataset to: {out_path}")
    print(f"Transitions: {len(actions)}")
    print(f"Positive reward transitions: {int(np.sum(np.asarray(rewards) > 0))}")
    print(f"Done transitions: {int(np.sum(dones))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect exhaustive MiniGrid transition data.")
    parser.add_argument("--env-id", default="MiniGrid-Empty-5x5-v0")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "datasets" / "minigrid_empty5_transitions.npz",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_transition_dataset(args.env_id, args.out, args.seed, args.repeats, args.gif)


if __name__ == "__main__":
    main()
