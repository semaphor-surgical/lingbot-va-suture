#!/usr/bin/env python3
"""Step 3: action normalization statistics (1st / 99th percentiles).

    python preprocess/compute_action_stats.py --dataset_path /path/to/suturebot_dataset

Training expresses the actions of a window relative to the pose at its start,
and a window can start at every latent frame of a segment. The statistics are
taken over the relative actions of all these windows. The result is printed in
the form of `norm_stat` in wan_va/configs/va_suturebot_cfg.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan_va.dataset.lerobot_latent_dataset import to_relative_actions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--target_fps", type=int, default=10,
                        help="Video frame rate used by extract_latents.py")
    args = parser.parse_args()
    root = Path(args.dataset_path)

    with open(root / "meta/info.json") as f:
        info = json.load(f)
    with open(root / "meta/episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    video_stride = round(info["fps"] / args.target_fps)  # recorded frames per video frame

    relative_actions = []
    for episode in tqdm(episodes):
        ep_idx = episode["episode_index"]
        chunk = ep_idx // info["chunks_size"]
        table = pq.read_table(
            root / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet",
            columns=["action"])
        actions = np.array(table.column("action").to_pylist())
        for seg in episode["action_config"]:
            segment = actions[seg["start_frame"]:seg["end_frame"]]
            if len(segment) < 6:  # too short for a training window
                continue
            num_latent_frames = (len(segment) // video_stride - 1) // 4 + 1
            for offset in range(num_latent_frames):
                start = offset * 4 * video_stride
                relative_actions.append(to_relative_actions(segment[start:]))

    relative_actions = np.concatenate(relative_actions, axis=0)
    stats = {
        "q01": np.percentile(relative_actions, 1, axis=0).tolist(),
        "q99": np.percentile(relative_actions, 99, axis=0).tolist(),
    }
    with open(root / "action_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {root / 'action_stats.json'}\n")
    print("norm_stat = {")
    for name in ("q01", "q99"):
        print(f'    "{name}": [')
        for i in range(0, 30, 7):
            print("        " + " ".join(f"{v:.6f}," for v in stats[name][i:i + 7]))
        print("    ],")
    print("}")


if __name__ == "__main__":
    main()
