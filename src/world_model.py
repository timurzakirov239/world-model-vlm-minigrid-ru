from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WorldModelConfig:
    state_dim: int = 5
    action_dim: int = 3
    hidden_dim: int = 128
    latent_dim: int = 32
    min_std: float = 0.1


class RSSMWorldModel(nn.Module):
    """Compact RSSM-style model for MiniGrid state transitions.

    It keeps a deterministic recurrent state h_t and a stochastic latent z_t.
    During training, posterior latents use the observed next state. During
    planning, the model rolls forward from its prior and decodes imagined states.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
        )
        self.initial_posterior = nn.Linear(config.hidden_dim, 2 * config.latent_dim)
        self.gru = nn.GRUCell(config.latent_dim + config.action_dim, config.hidden_dim)
        self.prior = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 2 * config.latent_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 2 * config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.latent_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
        )
        self.state_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.reward_head = nn.Linear(config.hidden_dim, 1)
        self.done_head = nn.Linear(config.hidden_dim, 1)

    def split_stats(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + self.config.min_std
        return mean, std

    def encode_initial(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.state_encoder(state)
        return self.split_stats(self.initial_posterior(encoded))

    def posterior_from_state(
        self,
        h: torch.Tensor,
        next_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.state_encoder(next_state)
        return self.split_stats(self.posterior(torch.cat([h, encoded], dim=-1)))

    def decode(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded = self.decoder(torch.cat([h, z], dim=-1))
        state = torch.sigmoid(self.state_head(decoded))
        reward = self.reward_head(decoded).squeeze(-1)
        done_logit = self.done_head(decoded).squeeze(-1)
        return state, reward, done_logit

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_one_hot = F.one_hot(action, self.config.action_dim).float()
        h_next = self.gru(torch.cat([z, action_one_hot], dim=-1), h)
        prior_mean, prior_std = self.split_stats(self.prior(h_next))
        return h_next, prior_mean, prior_std

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, horizon = actions.shape
        device = states.device
        h = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        z_mean, _ = self.encode_initial(states[:, 0])
        z = z_mean

        state_preds = []
        reward_preds = []
        done_logits = []
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []

        for t in range(horizon):
            h, prior_mean, prior_std = self.transition(h, z, actions[:, t])
            state_pred, reward_pred, done_logit = self.decode(h, prior_mean)
            post_mean, post_std = self.posterior_from_state(h, states[:, t + 1])
            z = post_mean

            state_preds.append(state_pred)
            reward_preds.append(reward_pred)
            done_logits.append(done_logit)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(post_mean)
            post_stds.append(post_std)

        return {
            "state": torch.stack(state_preds, dim=1),
            "reward": torch.stack(reward_preds, dim=1),
            "done_logit": torch.stack(done_logits, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_std": torch.stack(prior_stds, dim=1),
            "post_mean": torch.stack(post_means, dim=1),
            "post_std": torch.stack(post_stds, dim=1),
        }

    @torch.no_grad()
    def imagine(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if initial_state.ndim == 1:
            initial_state = initial_state.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        batch_size, horizon = actions.shape
        device = initial_state.device
        h = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        z, _ = self.encode_initial(initial_state)

        states = []
        rewards = []
        dones = []
        for t in range(horizon):
            h, prior_mean, _ = self.transition(h, z, actions[:, t])
            state_pred, reward_pred, done_logit = self.decode(h, prior_mean)
            z = prior_mean
            states.append(state_pred)
            rewards.append(reward_pred)
            dones.append(torch.sigmoid(done_logit))

        return {
            "state": torch.stack(states, dim=1),
            "reward": torch.stack(rewards, dim=1),
            "done_prob": torch.stack(dones, dim=1),
        }


def gaussian_kl(
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
) -> torch.Tensor:
    var_q = std_q.pow(2)
    var_p = std_p.pow(2)
    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p).pow(2)) / (2.0 * var_p) - 0.5
    return kl.sum(dim=-1)
