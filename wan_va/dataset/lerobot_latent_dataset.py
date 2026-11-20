# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified for LingBot-VA-Suture.
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from ..utils import logger

VAE_TEMPORAL_STRIDE = 4  # video frames per latent frame


def get_relative_pose(pose):
    """Express [N, 7] poses (xyz + xyzw quaternion) relative to the first one."""
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    relative_trans = pose[:, :3] - pose[0:1, :3]
    # canonical: qw >= 0, so that q and -q (the same rotation) give one target
    relative_quat = (first_rot.inv() * rot).as_quat(canonical=True)
    return np.concatenate([relative_trans, relative_quat], axis=1)


def to_relative_actions(action):
    """[N, 30] absolute actions -> both arm poses relative to the first row."""
    return np.concatenate(
        [
            get_relative_pose(action[:, 0:7]),  # left pose
            action[:, 7:14],
            get_relative_pose(action[:, 14:21]),  # right pose
            action[:, 21:],
        ],
        axis=1,
    )


class LatentLeRobotDataset(torch.utils.data.Dataset):
    """Training windows of a dataset built with the scripts in preprocess/.

    Every episode is cut into segments (`action_config` in meta/episodes.jsonl)
    whose video latents are stored in latents/. A training window starts at any
    latent frame of a segment and runs to the end of the segment; its actions
    are relative to the pose at the start of the window.

    Item:
        latents       [C, F, H, W * num_cameras]   the cameras side by side
        actions       [30, F, action_per_frame, 1] normalized to [-1, 1]
        actions_mask  same shape, True where an action is supervised
        text_emb      [512, 4096]
    """

    def __init__(self, config):
        self.root = Path(config.dataset_path)
        self.cam_keys = config.obs_cam_keys
        self.cfg_prob = config.cfg_prob
        self.action_per_frame = config.action_per_frame
        # recorded frames per video frame
        self.video_stride = config.action_per_frame // VAE_TEMPORAL_STRIDE
        self.channel_ids = config.inverse_used_action_channel_ids
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]

        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.task_text_embs = torch.load(self.root / 'task_text_embs.pt',
                                         weights_only=False)

        with open(self.root / 'meta' / 'info.json') as f:
            self.chunks_size = json.load(f)['chunks_size']
        with open(self.root / 'meta' / 'episodes.jsonl') as f:
            episodes = [json.loads(line) for line in f]

        latent_files = set((self.root / 'latents').glob('*/*/*.pth'))
        self.actions = {}  # episode index -> [T, 30] absolute actions
        self.windows = []  # (episode, start frame, end frame, text, latent offset)
        for episode in episodes:
            ep_idx = episode['episode_index']
            table = pq.read_table(self._episode_file(ep_idx),
                                  columns=['action'])
            self.actions[ep_idx] = np.array(table.column('action').to_pylist(),
                                            dtype=np.float32)
            for seg in episode['action_config']:
                start, end, text = seg['start_frame'], seg['end_frame'], seg[
                    'action_text']
                if text not in self.task_text_embs:
                    raise KeyError(f'task_text_embs.pt has no entry for "{text}"')
                if not all(
                        self._latent_file(key, ep_idx, start, end) in latent_files
                        for key in self.cam_keys):
                    continue  # latents not extracted
                num_video_frames = (end - start) // self.video_stride
                num_latent_frames = (num_video_frames -
                                     1) // VAE_TEMPORAL_STRIDE + 1
                # Every latent frame but the last starts a window. A window
                # from the last one would hold the look-back frame only,
                # without any supervised action.
                for offset in range(num_latent_frames - 1):
                    self.windows.append((ep_idx, start, end, text, offset))
        logger.info(f'Dataset {self.root}: {len(self.windows)} windows '
                    f'from {len(episodes)} episodes')

    def _episode_file(self, ep_idx):
        chunk = ep_idx // self.chunks_size
        return self.root / f'data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet'

    def _latent_file(self, cam_key, ep_idx, start, end):
        chunk = ep_idx // self.chunks_size
        return (self.root / f'latents/chunk-{chunk:03d}' / cam_key /
                f'episode_{ep_idx:06d}_{start}_{end}.pth')

    def __len__(self):
        return len(self.windows)

    def _process_actions(self, action, num_latent_frames):
        action = to_relative_actions(action)

        # Look-back rows in front of the window: identity pose, never supervised
        lookback = self.action_per_frame
        action = np.pad(action, ((lookback, 0), (0, 0)))
        action[:lookback, [6, 20]] = 1.0  # qw
        action = action[:num_latent_frames * self.action_per_frame]
        mask = np.ones_like(action, dtype='bool')
        mask[:lookback] = False

        # Append a zero column; the unused channels point to it
        action = np.pad(action, ((0, 0), (0, 1)))[:, self.channel_ids]
        mask = np.pad(mask, ((0, 0), (0, 1)))[:, self.channel_ids]

        action = (action - self.q01) / (self.q99 - self.q01 + 1e-6) * 2. - 1.
        action = np.clip(action, -1.5, 1.5)
        action = rearrange(action, '(f n) c -> c f n 1', f=num_latent_frames)
        mask = rearrange(mask, '(f n) c -> c f n 1', f=num_latent_frames)
        return torch.from_numpy(action * mask).float(), torch.from_numpy(mask)

    def __getitem__(self, idx):
        ep_idx, start, end, text, offset = self.windows[idx]

        latents = []
        for key in self.cam_keys:
            data = torch.load(self._latent_file(key, ep_idx, start, end),
                              weights_only=False)
            latent = rearrange(data['latent'],
                               '(f h w) c -> f h w c',
                               f=data['latent_num_frames'],
                               h=data['latent_height'],
                               w=data['latent_width'])
            latents.append(latent[offset:])
            if key == self.cam_keys[0]:
                frame_ids = data['frame_ids'][offset * VAE_TEMPORAL_STRIDE:]
        latents = torch.cat(latents, dim=2).permute(3, 0, 1, 2)

        # frame_ids: the recorded frames behind the video frames of the window
        num_latent_frames = (len(frame_ids) - 1) // VAE_TEMPORAL_STRIDE + 1
        actions, actions_mask = self._process_actions(
            self.actions[ep_idx][frame_ids[0]:end], num_latent_frames)

        text_emb = self.task_text_embs[text]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        return dict(
            latents=latents[:, :num_latent_frames],
            actions=actions,
            actions_mask=actions_mask,
            text_emb=text_emb,
        )
