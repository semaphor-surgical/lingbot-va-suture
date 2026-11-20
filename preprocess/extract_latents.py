#!/usr/bin/env python3
"""Step 2: encode the videos with the Wan2.2 VAE and the prompts with the text encoder.

    python preprocess/extract_latents.py --dataset_path /path/to/suturebot_dataset \
        --model_path /path/to/lingbot-va-base

Writes into the dataset directory
    latents/chunk-XXX/<camera>/episode_XXXXXX_<start>_<end>.pth   one per segment and camera
    task_text_embs.pt    {task prompt: text embedding}
    empty_emb.pt         embedding of the empty prompt (classifier-free guidance)

Videos are sampled at 10 fps and resized to 480 x 640 before encoding.
Segments that already have a latent file are skipped.
"""
import argparse
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel

# Long segments are encoded in pieces of 97 video frames (25 latent frames)
VAE_TEMPORAL_CHUNK = 97


def decode_frames(video_path, frame_ids):
    """Decode the given frames of a video as HxWx3 uint8 arrays."""
    frame_ids, frames = set(frame_ids), {}
    with av.open(str(video_path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for i, frame in enumerate(container.decode(video=0)):
            if i in frame_ids:
                frames[i] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(frame_ids):
                    break
    return frames


@torch.no_grad()
def encode_video(vae, frames, height, width, device, dtype):
    """List of HxWx3 uint8 frames -> normalized latents [C, F, H, W]."""
    video = torch.from_numpy(np.stack(frames)).to(device=device, dtype=torch.float32)
    video = video.permute(0, 3, 1, 2)
    if video.shape[2:] != (height, width):
        video = F.interpolate(video, size=(height, width), mode="bilinear",
                              align_corners=False)
    video = (video / 255.0 * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)

    latents = []
    for start in range(0, video.shape[2], VAE_TEMPORAL_CHUNK):
        chunk = video[:, :, start:start + VAE_TEMPORAL_CHUNK].to(dtype=dtype)
        mu = vae.encode(chunk).latent_dist.mean
        mean = torch.tensor(vae.config.latents_mean).to(mu.device, mu.dtype)
        std = torch.tensor(vae.config.latents_std).to(mu.device, mu.dtype)
        latents.append((mu - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1))
    return torch.cat(latents, dim=2)[0]


@torch.no_grad()
def encode_text(text_encoder, tokenizer, text, device, max_len=512):
    """Text -> [512, 4096] embedding, zero after the end of the text."""
    inputs = tokenizer([prompt_clean(text)], padding="max_length", max_length=max_len,
                       truncation=True, add_special_tokens=True,
                       return_attention_mask=True, return_tensors="pt")
    mask = inputs.attention_mask.to(device)
    emb = text_encoder(inputs.input_ids.to(device), mask).last_hidden_state[0]
    emb[int(mask.gt(0).sum()):] = 0
    return emb.to(torch.bfloat16).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True, help="lingbot-va-base")
    parser.add_argument("--target_fps", type=int, default=10)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video_workers", type=int, default=8)
    args = parser.parse_args()
    root, model_path = Path(args.dataset_path), Path(args.model_path)
    device, dtype = torch.device(args.device), torch.bfloat16

    with open(root / "meta/info.json") as f:
        info = json.load(f)
    with open(root / "meta/episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    cameras = sorted(k for k in info["features"] if "observation.images" in k)
    stride = max(1, round(info["fps"] / args.target_fps))

    # One job per video: the segments that still need a latent file
    jobs = []
    for cam in cameras:
        for episode in episodes:
            ep_idx = episode["episode_index"]
            chunk = ep_idx // info["chunks_size"]
            segments = []
            for seg in episode["action_config"]:
                start, end = seg["start_frame"], seg["end_frame"]
                latent_file = (root / f"latents/chunk-{chunk:03d}/{cam}" /
                               f"episode_{ep_idx:06d}_{start}_{end}.pth")
                if not latent_file.exists():
                    segments.append((start, end, latent_file))
            if segments:
                video = root / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep_idx:06d}.mp4"
                frame_ids = [i for s, e, _ in segments for i in range(s, e, stride)]
                jobs.append((video, frame_ids, segments))
    print(f"{len(jobs)} videos to encode")

    vae = AutoencoderKLWan.from_pretrained(str(model_path / "vae"), torch_dtype=dtype)
    vae = vae.to(device).eval()

    # Decode a few videos ahead in background threads while the VAE encodes
    with ThreadPoolExecutor(max_workers=args.video_workers) as pool:
        job_iter, pending = iter(jobs), deque()
        for _ in tqdm(range(len(jobs))):
            while len(pending) < args.video_workers:
                job = next(job_iter, None)
                if job is None:
                    break
                pending.append((job, pool.submit(decode_frames, job[0], job[1])))
            (video, _, segments), future = pending.popleft()
            frames = future.result()
            for start, end, latent_file in segments:
                frame_ids = [i for i in range(start, end, stride) if i in frames]
                latent = encode_video(vae, [frames[i] for i in frame_ids],
                                      args.height, args.width, device, dtype)
                latent_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "latent": latent.permute(1, 2, 3, 0).flatten(0, 2).to(torch.bfloat16).cpu(),
                    "latent_num_frames": latent.shape[1],
                    "latent_height": latent.shape[2],
                    "latent_width": latent.shape[3],
                    "frame_ids": frame_ids,
                    "start_frame": start,
                    "end_frame": end,
                    "fps": args.target_fps,
                    "ori_fps": info["fps"],
                }, latent_file)
    del vae
    torch.cuda.empty_cache()

    tokenizer = T5TokenizerFast.from_pretrained(str(model_path / "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(model_path / "text_encoder"), torch_dtype=dtype).to(device).eval()
    texts = sorted({seg["action_text"] for ep in episodes for seg in ep["action_config"]})
    torch.save({text: encode_text(text_encoder, tokenizer, text, device) for text in texts},
               root / "task_text_embs.pt")
    torch.save(encode_text(text_encoder, tokenizer, "", device), root / "empty_emb.pt")
    print(f"Text embeddings saved for {texts}")


if __name__ == "__main__":
    main()
