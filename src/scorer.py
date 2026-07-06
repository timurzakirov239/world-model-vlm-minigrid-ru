from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class StateScorer(Protocol):
    def score_states(self, states: np.ndarray) -> np.ndarray:
        ...


@dataclass
class GoalDistanceScorer:
    width: int
    height: int

    def score_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        agent_x = np.rint(np.clip(states[..., 0], 0.0, 1.0) * (self.width - 1))
        agent_y = np.rint(np.clip(states[..., 1], 0.0, 1.0) * (self.height - 1))
        goal_x = np.rint(np.clip(states[..., 3], 0.0, 1.0) * (self.width - 1))
        goal_y = np.rint(np.clip(states[..., 4], 0.0, 1.0) * (self.height - 1))
        distance = np.abs(agent_x - goal_x) + np.abs(agent_y - goal_y)
        max_distance = max((self.width - 2) + (self.height - 2), 1)
        return 1.0 - distance.astype(np.float32) / float(max_distance)


class CachedVLMScorer:
    def __init__(self, cache_path: Path):
        data = np.load(cache_path, allow_pickle=True)
        self.width = int(data["width"])
        self.height = int(data["height"])
        self.scores = data["scores"].astype(np.float32)
        self.goal_text = str(data["goal_text"])
        self.env_id = str(data["env_id"])

    def score_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        x = np.rint(np.clip(states[..., 0], 0.0, 1.0) * (self.width - 1)).astype(np.int64)
        y = np.rint(np.clip(states[..., 1], 0.0, 1.0) * (self.height - 1)).astype(np.int64)
        direction = np.rint(np.clip(states[..., 2], 0.0, 1.0) * 3.0).astype(np.int64) % 4
        return self.scores[x, y, direction]
