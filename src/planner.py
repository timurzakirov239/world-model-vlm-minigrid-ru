from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.scorer import CachedVLMScorer, GoalDistanceScorer, StateScorer
from src.world_model import RSSMWorldModel, WorldModelConfig


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 12
    candidates: int = 384
    gamma: float = 0.95
    vlm_weight: float = 0.8
    reward_weight: float = 1.0
    distance_weight: float = 0.0
    action_dim: int = 3


class RandomShootingPlanner:
    def __init__(
        self,
        model: RSSMWorldModel,
        config: PlannerConfig,
        scorer: StateScorer | None = None,
        distance_scorer: GoalDistanceScorer | None = None,
        seed: int = 0,
    ):
        self.model = model.eval()
        self.config = config
        self.scorer = scorer
        self.distance_scorer = distance_scorer
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def plan(self, state: np.ndarray, mode: str) -> tuple[int, dict[str, float]]:
        if mode == "random":
            return int(self.rng.integers(self.config.action_dim)), {"objective": 0.0}

        action_sequences = self.rng.integers(
            self.config.action_dim,
            size=(self.config.candidates, self.config.horizon),
            dtype=np.int64,
        )
        initial_states = np.repeat(state[None, :], self.config.candidates, axis=0)
        imagined = self.model.imagine(
            torch.from_numpy(initial_states.astype(np.float32)),
            torch.from_numpy(action_sequences),
        )
        imagined_states = imagined["state"].cpu().numpy()
        imagined_rewards = imagined["reward"].cpu().numpy()

        discounts = self.config.gamma ** np.arange(self.config.horizon, dtype=np.float32)
        reward_objective = (imagined_rewards * discounts[None, :]).sum(axis=1)
        objective = self.config.reward_weight * reward_objective

        vlm_objective = np.zeros_like(objective, dtype=np.float32)
        distance_objective = np.zeros_like(objective, dtype=np.float32)
        if mode == "wm_vlm":
            if self.scorer is not None:
                vlm_scores = self.scorer.score_states(imagined_states)
                vlm_objective = (vlm_scores * discounts[None, :]).sum(axis=1)
                objective = objective + self.config.vlm_weight * vlm_objective
            if self.distance_scorer is not None and self.config.distance_weight > 0:
                distance_scores = self.distance_scorer.score_states(imagined_states)
                distance_objective = (distance_scores * discounts[None, :]).sum(axis=1)
                objective = objective + self.config.distance_weight * distance_objective

        best_idx = int(np.argmax(objective))
        return int(action_sequences[best_idx, 0]), {
            "objective": float(objective[best_idx]),
            "reward_objective": float(reward_objective[best_idx]),
            "vlm_objective": float(vlm_objective[best_idx]),
            "distance_objective": float(distance_objective[best_idx]),
        }


def load_world_model(checkpoint_path: Path) -> RSSMWorldModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = WorldModelConfig(**checkpoint["config"])
    model = RSSMWorldModel(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_vlm_scorer(cache_path: Path | None) -> CachedVLMScorer | None:
    if cache_path is None:
        return None
    return CachedVLMScorer(cache_path)
