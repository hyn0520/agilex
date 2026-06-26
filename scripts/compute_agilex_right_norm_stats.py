"""Compute norm stats for the AgileX right-arm dataset without decoding videos."""

from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from openpi.shared import normalize
from openpi.training import config as _config


def main() -> None:
    config = _config.get_config("pi05_agilex_right_finetune")
    dataset_dir = Path(config.data.repo_id)
    asset_id = config.data.assets.asset_id
    if asset_id is None:
        raise ValueError("Expected pi05_agilex_right_finetune to define an asset_id")
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir / 'data'}")

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for parquet_file in tqdm.tqdm(parquet_files, desc="Computing AgileX right-arm stats"):
        df = pd.read_parquet(parquet_file, columns=["observation.state", "action"])
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)[:, 7:14]
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)[:, 7:14]
        stats["state"].update(states)
        stats["actions"].update(actions)

    output_path = config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, {key: value.get_statistics() for key, value in stats.items()})


if __name__ == "__main__":
    main()
