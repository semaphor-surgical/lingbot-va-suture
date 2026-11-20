#!/usr/bin/env python3
"""Step 1: convert SutureBot (Open-H-Embodiment, LeRobot v2) to the training layout.

    python preprocess/prepare_dataset.py --input_dir /path/to/suturebot \
        --output_dir /path/to/suturebot_dataset

For every episode this
  - maps the 16-D dVRK action [PSM1 pose, PSM1 jaw, PSM2 pose, PSM2 jaw]
    to the 30-D action layout of LingBot-VA (see wan_va/configs/va_suturebot_cfg.py),
  - links the three camera videos under the camera names of the model,
  - cuts the episode into segments of at most 10 s, labelled with its sub-task.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MAX_SEGMENT_FRAMES = 300  # 10 s at 30 fps

CAMERA_MAP = {
    "observation.images.endoscope.left": "observation.images.cam_high",
    "observation.images.wrist.left": "observation.images.cam_left_wrist",
    "observation.images.wrist.right": "observation.images.cam_right_wrist",
}

ACTION_NAMES = (
    [f"{n}_left" for n in ("x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w")]
    + [f"joint_left_{i}" for i in range(7)]
    + [f"{n}_right" for n in ("x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w")]
    + [f"joint_right_{i}" for i in range(7)]
    + ["jaw_left", "jaw_right"]
)


def to_30d(action):
    """dVRK [PSM1 pose (7), PSM1 jaw, PSM2 pose (7), PSM2 jaw] -> 30-D layout.

    PSM2 is the left arm, PSM1 the right arm. The joint channels stay zero.
    """
    out = np.zeros((action.shape[0], 30), dtype=np.float64)
    out[:, 0:7] = action[:, 8:15]  # left pose
    out[:, 28] = action[:, 15]  # left jaw
    out[:, 14:21] = action[:, 0:7]  # right pose
    out[:, 29] = action[:, 7]  # right jaw
    return out


def process_episode(job):
    src, dst, episode, chunk = job["src"], job["dst"], job["episode"], job["chunk"]
    ep_idx = episode["episode_index"]
    name = f"chunk-{chunk:03d}/episode_{ep_idx:06d}"

    table = pq.read_table(src / f"data/{name}.parquet")
    action = to_30d(np.array(table.column("action").to_pylist()))
    length = action.shape[0]
    quat_norms = np.linalg.norm(action[:, [3, 4, 5, 6, 17, 18, 19, 20]].reshape(
        length, 2, 4), axis=2)
    if length < 10 or np.any(np.abs(quat_norms - 1) > 0.05):
        return None  # too short, or a broken quaternion

    out_file = dst / f"data/{name}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "action": action.tolist(),
            "timestamp": table.column("timestamp").to_pylist(),
            "frame_index": list(range(length)),
            "episode_index": [ep_idx] * length,
            "index": list(range(length)),
            "task_index": table.column("task_index").to_pylist(),
        }), out_file)

    for src_cam, dst_cam in CAMERA_MAP.items():
        video = dst / f"videos/chunk-{chunk:03d}/{dst_cam}/episode_{ep_idx:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        if not video.exists():
            os.symlink(
                os.path.realpath(
                    src / f"videos/chunk-{chunk:03d}/{src_cam}/episode_{ep_idx:06d}.mp4"),
                video)

    tasks = [t.replace("_", " ").lower() for t in episode["tasks"]]
    return {
        "episode_index": ep_idx,
        "tasks": tasks,
        "length": length,
        "action_config": [{
            "start_frame": start,
            "end_frame": min(start + MAX_SEGMENT_FRAMES, length),
            "action_text": tasks[0],
        } for start in range(0, length, MAX_SEGMENT_FRAMES)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    src, dst = Path(args.input_dir), Path(args.output_dir)

    with open(src / "meta/info.json") as f:
        info = json.load(f)
    with open(src / "meta/episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    with open(src / "meta/tasks.jsonl") as f:
        tasks = [json.loads(line) for line in f]
    for task in tasks:
        task["task"] = task["task"].replace("_", " ").lower()

    jobs = [{
        "src": src,
        "dst": dst,
        "episode": episode,
        "chunk": episode["episode_index"] // info["chunks_size"],
    } for episode in episodes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            tqdm(pool.map(process_episode, jobs, chunksize=8), total=len(jobs)))
    episodes_out = [r for r in results if r is not None]

    features = {
        dst_cam: info["features"][src_cam]
        for src_cam, dst_cam in CAMERA_MAP.items()
    }
    features["action"] = {"dtype": "float64", "shape": [30], "names": ACTION_NAMES}
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = info["features"][key]
    info_out = {
        "codebase_version": "v2.0",
        "robot_type": info["robot_type"],
        "total_episodes": len(episodes_out),
        "total_frames": sum(ep["length"] for ep in episodes_out),
        "total_tasks": len(tasks),
        "chunks_size": info["chunks_size"],
        "fps": info["fps"],
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "features": features,
    }

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    with open(dst / "meta/info.json", "w") as f:
        json.dump(info_out, f, indent=2)
    with open(dst / "meta/episodes.jsonl", "w") as f:
        for episode in episodes_out:
            f.write(json.dumps(episode) + "\n")
    with open(dst / "meta/tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"{len(episodes_out)} of {len(episodes)} episodes written to {dst}")


if __name__ == "__main__":
    main()
