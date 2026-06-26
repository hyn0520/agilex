#!/usr/bin/env python3
"""Offline sanity check for an AgileX policy checkpoint/server.

This script samples observations from a LeRobot dataset and queries either an
already-running websocket policy server or a checkpoint loaded in-process. It
does not command a robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from openpi_client import image_tools
from openpi_client import websocket_client_policy


def _load_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import av

    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Frame {frame_index} not found in {video_path}")


def _episode_file(dataset_dir: Path, episode_index: int) -> Path:
    return dataset_dir / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def _video_file(dataset_dir: Path, episode_index: int, video_key: str) -> Path:
    return dataset_dir / f"videos/chunk-{episode_index // 1000:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def load_observation(dataset_dir: Path, episode_index: int, frame_index: int, prompt: str) -> dict:
    table = pq.read_table(_episode_file(dataset_dir, episode_index))
    data = table.to_pydict()
    if frame_index < 0:
        frame_index = len(data["frame_index"]) + frame_index
    if frame_index < 0 or frame_index >= len(data["frame_index"]):
        raise IndexError(f"{frame_index=} out of range for episode with {len(data['frame_index'])} frames")

    front = _load_video_frame(_video_file(dataset_dir, episode_index, "observation.images.front"), frame_index)
    right = _load_video_frame(_video_file(dataset_dir, episode_index, "observation.images.right"), frame_index)
    state = np.asarray(data["observation.state"][frame_index], dtype=np.float32)

    return {
        "observation/images/front": image_tools.convert_to_uint8(image_tools.resize_with_pad(front, 224, 224)),
        "observation/images/right": image_tools.convert_to_uint8(image_tools.resize_with_pad(right, 224, 224)),
        "observation/state": state,
        "prompt": prompt,
    }


def create_policy(args: argparse.Namespace):
    if args.checkpoint_dir is None:
        return websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    from openpi.policies import policy_config
    from openpi.training import config as train_config

    return policy_config.create_trained_policy(
        train_config.get_config(args.config),
        args.checkpoint_dir,
        default_prompt=args.prompt,
    )


def print_actions(actions: np.ndarray, *, label: str) -> None:
    print(f"\n## {label}")
    print("shape:", actions.shape, "dtype:", actions.dtype)
    print("first 5 rows:")
    np.set_printoptions(precision=5, suppress=True)
    print(actions[:5])
    print("per-dim min:", actions.min(axis=0))
    print("per-dim max:", actions.max(axis=0))
    print("per-dim mean:", actions.mean(axis=0))
    print("gripper min/max/mean:", float(actions[:, 6].min()), float(actions[:, 6].max()), float(actions[:, 6].mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("/mnt/data/real_data/stack_bowl_0617_v21"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--prompt", default="put one bowl onto another one")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--config", default="pi05_agilex_right_finetune")
    args = parser.parse_args()

    obs = load_observation(args.dataset_dir, args.episode_index, args.frame_index, args.prompt)
    print("Loaded offline observation:")
    print("  front:", obs["observation/images/front"].shape, obs["observation/images/front"].dtype)
    print("  right:", obs["observation/images/right"].shape, obs["observation/images/right"].dtype)
    print("  state:", obs["observation/state"].shape, obs["observation/state"].dtype)
    print("  right state:", obs["observation/state"][7:14])
    print("  prompt:", obs["prompt"])

    policy = create_policy(args)
    start = time.monotonic()
    result = policy.infer(obs)
    elapsed_ms = (time.monotonic() - start) * 1000

    actions = np.asarray(result["actions"])
    print_actions(actions, label=f"policy actions ({elapsed_ms:.1f} ms)")
    if "policy_timing" in result:
        print("policy_timing:", result["policy_timing"])


if __name__ == "__main__":
    main()
