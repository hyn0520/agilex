#!/usr/bin/env python3
"""Convert local LeRobot v3-style AgileX datasets to the v2.1 layout.

This converter is intentionally conservative: it does not transcode videos or
rewrite parquet payloads. It relinks/copies files into the v2.1 path layout and
generates the jsonl metadata files expected by the LeRobot version used in this
repo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm

try:
    import av
except ImportError:  # pragma: no cover - script still works without video probing.
    av = None


DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def _append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, default=_json_default) + "\n")


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination file already exists: {dst}")

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_tasks(src: Path) -> dict[int, str]:
    tasks_path = src / "meta/tasks.parquet"
    table = pq.read_table(tasks_path).to_pydict()
    return {int(i): str(task) for i, task in zip(table["task_index"], table["task"])}


def _numeric_feature_stats(df: pd.DataFrame, features: dict) -> dict:
    stats = {}
    for key in df.columns:
        feature = features.get(key)
        if feature is None or feature["dtype"] in {"video", "image", "string"}:
            continue

        values = df[key].to_numpy()
        first = values[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            array = np.stack(values).astype(np.float64)
            keepdims = False
        else:
            array = values.astype(np.float64)
            keepdims = True

        stats[key] = {
            "min": np.min(array, axis=0, keepdims=keepdims),
            "max": np.max(array, axis=0, keepdims=keepdims),
            "mean": np.mean(array, axis=0, keepdims=keepdims),
            "std": np.std(array, axis=0, keepdims=keepdims),
            "count": np.array([len(array)], dtype=np.int64),
        }
    return stats


def _aggregate_feature_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([x["mean"] for x in items])
    variances = np.stack([x["std"] ** 2 for x in items])
    counts = np.stack([x["count"] for x in items])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    total_mean = (means * counts).sum(axis=0) / total_count
    weighted_variances = (variances + (means - total_mean) ** 2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([x["min"] for x in items]), axis=0),
        "max": np.max(np.stack([x["max"] for x in items]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def _aggregate_stats(episode_stats: list[dict]) -> dict:
    keys = sorted({key for stats in episode_stats for key in stats})
    return {key: _aggregate_feature_stats([stats[key] for stats in episode_stats if key in stats]) for key in keys}


def _probe_video_info(path: Path, fps: int) -> tuple[list[int], dict] | None:
    if av is None:
        return None
    try:
        with av.open(str(path), "r") as video_file:
            stream = video_file.streams.video[0]
            channels = 3
            info = {
                "video.height": stream.height,
                "video.width": stream.width,
                "video.codec": stream.codec_context.name,
                "video.pix_fmt": stream.pix_fmt,
                "video.is_depth_map": False,
                "video.fps": int(stream.base_rate) if stream.base_rate is not None else fps,
                "video.channels": channels,
                "has_audio": len(video_file.streams.audio) > 0,
            }
            return [channels, stream.height, stream.width], info
    except Exception:
        return None


def _convert_info(src: Path, src_info: dict, *, total_episodes: int, total_frames: int) -> dict:
    features = dict(src_info["features"])
    converted_features = {}
    for key, feature in features.items():
        feature = dict(feature)
        if feature["dtype"] == "video":
            src_video = src / src_info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
            if not src_video.exists():
                src_video = src / f"videos/{key}/chunk-000/file-000.mp4"
            probed = _probe_video_info(src_video, src_info["fps"]) if src_video.exists() else None
            if probed is not None:
                feature["shape"], feature["info"] = probed
                feature["names"] = ["channels", "height", "width"]
                converted_features[key] = feature
                continue

            shape = feature.get("shape", [])
            if len(shape) == 3 and shape[-1] == 3:
                height, width, channels = shape
                feature["shape"] = [channels, height, width]
                feature["names"] = ["channels", "height", "width"]
                feature.setdefault(
                    "info",
                    {
                        "video.height": height,
                        "video.width": width,
                        "video.codec": "unknown",
                        "video.pix_fmt": "unknown",
                        "video.is_depth_map": False,
                        "video.fps": src_info["fps"],
                        "video.channels": channels,
                        "has_audio": False,
                    },
                )
        converted_features[key] = feature

    return {
        **src_info,
        "codebase_version": "v2.1",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": converted_features,
    }


def convert(src: Path, dst: Path, *, mode: str) -> None:
    if not (src / "meta/info.json").exists():
        raise FileNotFoundError(f"Missing source info.json: {src / 'meta/info.json'}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists, refusing to overwrite: {dst}")

    src_info = json.loads((src / "meta/info.json").read_text())
    tasks = _load_tasks(src)
    data_files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No v3 data files found under {src / 'data'}")

    dst.mkdir(parents=True)
    for task_index, task in sorted(tasks.items()):
        _append_jsonl(dst / "meta/tasks.jsonl", {"task_index": task_index, "task": task})

    all_episode_stats = []
    total_frames = 0
    source_manifest = json.loads((src / "build_manifest.json").read_text()) if (src / "build_manifest.json").exists() else {}
    source_episodes = {int(ep["episode_index"]): ep for ep in source_manifest.get("source_episodes", [])}

    for ep_index, parquet_file in enumerate(tqdm.tqdm(data_files, desc="Converting episodes")):
        episode_chunk = ep_index // int(src_info.get("chunks_size", 1000))
        dst_parquet = dst / DEFAULT_PARQUET_PATH.format(episode_chunk=episode_chunk, episode_index=ep_index)
        _link_or_copy(parquet_file, dst_parquet, mode=mode)

        df = pd.read_parquet(parquet_file)
        length = len(df)
        total_frames += length
        task_indices = sorted({int(x) for x in df["task_index"].unique()})
        episode_tasks = [tasks[i] for i in task_indices]

        _append_jsonl(
            dst / "meta/episodes.jsonl",
            {
                "episode_index": ep_index,
                "tasks": episode_tasks,
                "length": length,
            },
        )

        ep_stats = _numeric_feature_stats(df, src_info["features"])
        all_episode_stats.append(ep_stats)
        _append_jsonl(dst / "meta/episodes_stats.jsonl", {"episode_index": ep_index, "stats": ep_stats})

        if ep_index in source_episodes:
            _append_jsonl(dst / "meta/source_episodes.jsonl", source_episodes[ep_index])

        for video_key in [key for key, ft in src_info["features"].items() if ft["dtype"] == "video"]:
            src_video = src / src_info["video_path"].format(
                video_key=video_key,
                chunk_index=0,
                file_index=ep_index,
            )
            if not src_video.exists():
                # Fall back to the common one-chunk v3 path used by these datasets.
                src_video = src / f"videos/{video_key}/chunk-000/file-{ep_index:03d}.mp4"
            if not src_video.exists():
                raise FileNotFoundError(f"Missing source video for episode {ep_index}: {video_key}")

            dst_video = dst / DEFAULT_VIDEO_PATH.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=ep_index,
            )
            _link_or_copy(src_video, dst_video, mode=mode)

    dst_info = _convert_info(src, src_info, total_episodes=len(data_files), total_frames=total_frames)
    _write_json(dst / "meta/info.json", dst_info)
    _write_json(dst / "meta/stats.json", _aggregate_stats(all_episode_stats))

    if (src / "build_manifest.json").exists():
        _link_or_copy(src / "build_manifest.json", dst / "build_manifest.json", mode=mode)

    print(f"Converted {len(data_files)} episodes / {total_frames} frames")
    print(f"Wrote v2.1 dataset to: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Source LeRobot v3 dataset directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination LeRobot v2.1 dataset directory")
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place parquet/video payloads in the destination. hardlink falls back to copy.",
    )
    args = parser.parse_args()
    convert(args.src.resolve(), args.dst.resolve(), mode=args.mode)


if __name__ == "__main__":
    main()
