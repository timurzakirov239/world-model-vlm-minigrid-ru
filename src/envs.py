from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments in Gymnasium
import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    image_size: int = 56


def make_minigrid_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    return gym.make(env_id, render_mode=render_mode)


def observation_to_array(obs: dict[str, Any], env: gym.Env) -> np.ndarray:
    """Convert MiniGrid symbolic observation into a small numeric image tensor."""
    image = obs["image"].astype(np.float32)

    object_idx = image[:, :, 0] / 10.0
    color_idx = image[:, :, 1] / 10.0
    state_idx = image[:, :, 2] / 10.0

    direction = np.full_like(object_idx, float(obs["direction"]) / 3.0)
    stacked = np.stack([object_idx, color_idx, state_idx, direction], axis=0)
    return stacked.astype(np.float32)


def render_rgb(env: gym.Env) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("Environment did not return an RGB frame. Use render_mode='rgb_array'.")
    return frame


def get_goal_pos(env: gym.Env) -> tuple[int, int]:
    unwrapped = env.unwrapped
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "goal":
                return int(x), int(y)
    raise RuntimeError("Could not find a goal cell in the MiniGrid environment.")


def get_raw_state(env: gym.Env) -> np.ndarray:
    unwrapped = env.unwrapped
    goal_x, goal_y = get_goal_pos(env)
    agent_x, agent_y = unwrapped.agent_pos
    return np.asarray(
        [agent_x, agent_y, unwrapped.agent_dir, goal_x, goal_y],
        dtype=np.float32,
    )


def normalize_state(raw_state: np.ndarray, width: int, height: int) -> np.ndarray:
    state = raw_state.astype(np.float32).copy()
    state[0] /= max(width - 1, 1)
    state[1] /= max(height - 1, 1)
    state[2] /= 3.0
    state[3] /= max(width - 1, 1)
    state[4] /= max(height - 1, 1)
    return state


def denormalize_state(state: np.ndarray, width: int, height: int) -> np.ndarray:
    raw = state.astype(np.float32).copy()
    raw[0] = np.rint(np.clip(raw[0], 0.0, 1.0) * max(width - 1, 1))
    raw[1] = np.rint(np.clip(raw[1], 0.0, 1.0) * max(height - 1, 1))
    raw[2] = np.rint(np.clip(raw[2], 0.0, 1.0) * 3.0)
    raw[3] = np.rint(np.clip(raw[3], 0.0, 1.0) * max(width - 1, 1))
    raw[4] = np.rint(np.clip(raw[4], 0.0, 1.0) * max(height - 1, 1))
    return raw.astype(np.int64)


def state_from_env(env: gym.Env) -> np.ndarray:
    unwrapped = env.unwrapped
    return normalize_state(get_raw_state(env), unwrapped.width, unwrapped.height)


def set_raw_state(env: gym.Env, raw_state: np.ndarray, step_count: int = 0) -> None:
    unwrapped = env.unwrapped
    raw = raw_state.astype(np.int64)
    unwrapped.agent_pos = (int(raw[0]), int(raw[1]))
    unwrapped.agent_dir = int(raw[2] % 4)
    unwrapped.step_count = int(step_count)


def valid_empty_positions(env: gym.Env) -> list[tuple[int, int]]:
    unwrapped = env.unwrapped
    positions: list[tuple[int, int]] = []
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is None or getattr(cell, "type", None) == "goal":
                positions.append((int(x), int(y)))
    return positions


def render_state(env_id: str, state: np.ndarray, seed: int = 0) -> np.ndarray:
    env = make_minigrid_env(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    raw = denormalize_state(state, env.unwrapped.width, env.unwrapped.height)
    set_raw_state(env, raw)
    frame = render_rgb(env)
    env.close()
    return frame
