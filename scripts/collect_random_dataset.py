from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs import make_minigrid_env, observation_to_array, render_rgb  # noqa: E402


def collect_dataset(
    env_id: str,
    episodes: int,
    max_steps: int,
    seed: int,
    out_path: Path,
    gif_episodes: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_dir = PROJECT_ROOT / "artifacts" / "dataset_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    env = make_minigrid_env(env_id, render_mode="rgb_array")
    rng = np.random.default_rng(seed)

    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []
    timesteps = []
    episode_returns = []
    episode_lengths = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        frames = [render_rgb(env)]
        episode_return = 0.0

        for timestep in range(max_steps):
            action = int(rng.integers(env.action_space.n))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            observations.append(observation_to_array(obs, env))
            next_observations.append(observation_to_array(next_obs, env))
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done)
            episode_ids.append(episode)
            timesteps.append(timestep)

            episode_return += float(reward)
            frames.append(render_rgb(env))
            obs = next_obs

            if done:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(len(frames) - 1)

        if episode < gif_episodes:
            safe_env = env_id.replace("/", "_").replace(":", "_")
            imageio.mimsave(
                gif_dir / f"{safe_env}_random_episode_{episode:03d}.gif",
                frames,
                duration=0.18,
            )

    env.close()

    np.savez_compressed(
        out_path,
        observations=np.asarray(observations, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        env_id=np.asarray(env_id),
        seed=np.asarray(seed, dtype=np.int64),
        max_steps=np.asarray(max_steps, dtype=np.int64),
    )

    success_rate = float(np.mean(np.asarray(episode_returns) > 0.0))
    print(f"Saved dataset to: {out_path}")
    print(f"Transitions: {len(actions)}")
    print(f"Episodes: {episodes}")
    print(f"Mean return: {np.mean(episode_returns):.4f}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect random MiniGrid trajectories.")
    parser.add_argument("--env-id", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "datasets" / "minigrid_empty_random_200.npz",
    )
    parser.add_argument("--gif-episodes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_dataset(
        env_id=args.env_id,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        out_path=args.out,
        gif_episodes=args.gif_episodes,
    )


if __name__ == "__main__":
    main()
