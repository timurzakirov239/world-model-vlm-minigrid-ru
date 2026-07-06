from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs import (  # noqa: E402
    make_minigrid_env,
    normalize_state,
    render_rgb,
    set_raw_state,
    valid_empty_positions,
)


def load_clip(model_name: str, pretrained: str, cache_dir: Path):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device="cpu",
        cache_dir=str(cache_dir),
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def compute_vlm_cache(
    env_id: str,
    goal_text: str,
    out_path: Path,
    cache_dir: Path,
    model_name: str,
    pretrained: str,
    seed: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = make_minigrid_env(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    width, height = env.unwrapped.width, env.unwrapped.height

    states = []
    images = []
    raw_states = []
    for x, y in valid_empty_positions(env):
        for direction in range(4):
            raw = np.asarray([x, y, direction, width - 2, height - 2], dtype=np.float32)
            set_raw_state(env, raw)
            frame = render_rgb(env)
            raw_states.append(raw)
            states.append(normalize_state(raw, width, height))
            images.append(Image.fromarray(frame))
    env.close()

    print(f"Rendering {len(images)} possible states for VLM scoring.")
    model, preprocess, tokenizer = load_clip(model_name, pretrained, cache_dir)

    with torch.no_grad():
        image_batch = torch.stack([preprocess(image) for image in images])
        text = tokenizer([goal_text])
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        raw_scores = (image_features @ text_features.T).squeeze(-1).float().cpu().numpy()

    scores_flat = (raw_scores - raw_scores.min()) / max(raw_scores.max() - raw_scores.min(), 1e-6)
    score_grid = np.full((width, height, 4), -1.0, dtype=np.float32)
    for raw, score in zip(raw_states, scores_flat):
        x, y, direction = raw[:3].astype(int)
        score_grid[x, y, direction] = float(score)

    best_idx = int(np.argmax(scores_flat))
    best_raw = raw_states[best_idx].astype(int).tolist()
    np.savez_compressed(
        out_path,
        scores=score_grid,
        states=np.asarray(states, dtype=np.float32),
        raw_states=np.asarray(raw_states, dtype=np.float32),
        raw_scores=raw_scores.astype(np.float32),
        normalized_scores=scores_flat.astype(np.float32),
        goal_text=np.asarray(goal_text),
        env_id=np.asarray(env_id),
        model_name=np.asarray(model_name),
        pretrained=np.asarray(pretrained),
        width=np.asarray(width, dtype=np.int64),
        height=np.asarray(height, dtype=np.int64),
        best_raw_state=np.asarray(best_raw, dtype=np.int64),
    )
    print(f"Saved VLM cache to: {out_path}")
    print(f"Best scored raw state [x, y, dir, goal_x, goal_y]: {best_raw}")
    print(f"Score range: raw=({raw_scores.min():.4f}, {raw_scores.max():.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute VLM scores for all MiniGrid states.")
    parser.add_argument("--env-id", default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--goal-text", default="agent at the green goal")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "vlm_cache" / "empty5_clip_goal.npz",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "model_cache" / "open_clip",
    )
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_vlm_cache(
        env_id=args.env_id,
        goal_text=args.goal_text,
        out_path=args.out,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
