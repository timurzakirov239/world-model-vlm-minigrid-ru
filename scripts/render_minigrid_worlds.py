from __future__ import annotations

import random
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import minigrid  # noqa: F401 - registers MiniGrid environments in Gymnasium


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "artifacts" / "minigrid_samples"


def render_env(env_id: str, seed: int = 0, steps: int = 40) -> None:
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames = [env.render()]
    total_reward = 0.0
    terminated = False
    truncated = False

    rng = random.Random(seed)
    for _ in range(steps):
        action = rng.randrange(env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()

    safe_name = env_id.replace("/", "_").replace(":", "_")
    imageio.imwrite(OUT_DIR / f"{safe_name}_start.png", frames[0])
    imageio.mimsave(OUT_DIR / f"{safe_name}_random.gif", frames, duration=0.18)

    print(
        f"{env_id}: saved {len(frames)} frames, "
        f"total_reward={total_reward:.3f}, "
        f"terminated={terminated}, truncated={truncated}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_ids = [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-DoorKey-8x8-v0",
    ]

    for env_id in env_ids:
        render_env(env_id)

    print(f"Saved samples to: {OUT_DIR}")


if __name__ == "__main__":
    main()
