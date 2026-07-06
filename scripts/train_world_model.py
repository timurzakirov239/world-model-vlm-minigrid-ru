from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model import RSSMWorldModel, WorldModelConfig, gaussian_kl  # noqa: E402


class TransitionSequenceDataset(Dataset):
    def __init__(self, npz_path: Path, seq_len: int):
        data = np.load(npz_path, allow_pickle=True)
        states = data["states"].astype(np.float32)
        next_states = data["next_states"].astype(np.float32)
        actions = data["actions"].astype(np.int64)
        rewards = data["rewards"].astype(np.float32)
        dones = data["dones"].astype(np.float32)

        # The exhaustive dataset stores independent transitions. For an RSSM we
        # train short one-step sequences; the recurrent state is still used, and
        # imagined rollouts chain the learned prior at planning time.
        self.states = np.stack([states, next_states], axis=1)
        self.actions = actions[:, None]
        self.rewards = rewards[:, None]
        self.dones = dones[:, None]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.from_numpy(self.states[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
        }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TransitionSequenceDataset(args.data, seq_len=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = WorldModelConfig(action_dim=args.action_dim)
    model = RSSMWorldModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "state": 0.0,
            "reward": 0.0,
            "done": 0.0,
            "kl": 0.0,
        }
        count = 0

        for batch in loader:
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]

            out = model(states, actions)
            state_loss = F.mse_loss(out["state"], states[:, 1:])
            reward_loss = F.mse_loss(out["reward"], rewards)
            done_loss = F.binary_cross_entropy_with_logits(out["done_logit"], dones)
            kl_loss = gaussian_kl(
                out["post_mean"],
                out["post_std"],
                out["prior_mean"],
                out["prior_std"],
            ).mean()
            loss = (
                args.state_weight * state_loss
                + args.reward_weight * reward_loss
                + args.done_weight * done_loss
                + args.kl_weight * kl_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()

            batch_size = states.shape[0]
            totals["loss"] += float(loss.item()) * batch_size
            totals["state"] += float(state_loss.item()) * batch_size
            totals["reward"] += float(reward_loss.item()) * batch_size
            totals["done"] += float(done_loss.item()) * batch_size
            totals["kl"] += float(kl_loss.item()) * batch_size
            count += batch_size

        metrics = {key: value / count for key, value in totals.items()}
        metrics["epoch"] = epoch
        history.append(metrics)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={metrics['loss']:.5f} "
                f"state={metrics['state']:.5f} reward={metrics['reward']:.5f} "
                f"done={metrics['done']:.5f} kl={metrics['kl']:.5f}"
            )

    checkpoint = {
        "model": model.state_dict(),
        "config": config.__dict__,
        "action_dim": args.action_dim,
        "data": str(args.data),
        "history": history,
        "seed": args.seed,
    }
    ckpt_path = args.out_dir / "rssm_world_model.pt"
    torch.save(checkpoint, ckpt_path)

    history_path = args.out_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Saved history to: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact RSSM world model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "datasets" / "minigrid_empty5_transitions.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "empty5_rssm",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-weight", type=float, default=10.0)
    parser.add_argument("--reward-weight", type=float, default=5.0)
    parser.add_argument("--done-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
