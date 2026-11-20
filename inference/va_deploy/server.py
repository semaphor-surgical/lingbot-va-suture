#!/usr/bin/env python3
"""LingBot-VA-Suture inference server.

Closed loop (one session, the KV cache is kept between calls):
    POST /reset        start a session with a task prompt
    POST /infer_chunk  predict the next chunk of actions
    POST /cache        send the frames observed while that chunk was executed
    ... repeat /infer_chunk and /cache

Open loop:
    POST /infer        predicted video and actions from a single observation

Images are base64-encoded PNGs, one per camera, in the order that
GET /health reports under "cameras".

Usage:
    python inference/va_deploy/server.py --config deploy_config.yaml
"""
import argparse
import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
import yaml
from easydict import EasyDict
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from wan_va.configs import VA_CONFIGS  # noqa: E402
from wan_va.wan_va_server import VA_Server  # noqa: E402

# Keys of the model config that the deploy config may override
MODEL_OVERRIDES = ('enable_offload', 'num_inference_steps',
                   'action_num_inference_steps', 'video_exec_step',
                   'guidance_scale', 'action_guidance_scale')


class ResetRequest(BaseModel):
    prompt: str = Field(..., description='Task prompt of the session.')


class ChunkRequest(BaseModel):
    frames: list[str] | None = Field(
        None,
        description='Current observation, one image per camera. '
        'Required for the first chunk of a session.')


class CacheRequest(BaseModel):
    frames: list[list[str]] = Field(
        ...,
        description='Frames observed at 10 fps while the last chunk was '
        'executed: a list over time, each entry one image per camera.')


class InferRequest(BaseModel):
    frames: list[str] = Field(...,
                              description='Observation, one image per camera.')
    text: str = Field(..., description='Task prompt.')
    num_chunks: int = Field(4, ge=1, le=16)


def decode_image(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB'))


def encode_image(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_model(deploy_cfg):
    config = EasyDict(VA_CONFIGS[deploy_cfg['va_config']])
    config.wan22_pretrained_model_name_or_path = deploy_cfg['base_model']
    config.checkpoint_path = deploy_cfg['checkpoint']
    for key in MODEL_OVERRIDES:
        if key in deploy_cfg:
            config[key] = deploy_cfg[key]
            print(f'  {key} = {deploy_cfg[key]} (deploy config)')

    model = VA_Server(config)

    if deploy_cfg.get('torch_compile', False):
        print('Compiling the transformer (the first inference is slow) ...')
        model.transformer = torch.compile(model.transformer,
                                          mode='default',
                                          fullgraph=False)
        t0 = time.time()
        blank = {
            key: np.zeros((config.height, config.width, 3), dtype=np.uint8)
            for key in config.obs_cam_keys
        }
        model.reset(config.task_prompts[0])
        model.infer_chunk(blank)
        model.reset(config.task_prompts[0])
        print(f'Warm-up done ({time.time() - t0:.1f}s)')

    print(f"Model ready: {deploy_cfg['checkpoint']}")
    return model


def create_app(deploy_cfg):
    state = {}
    seed = deploy_cfg.get('seed', 42) if deploy_cfg.get('deterministic',
                                                         True) else None

    @asynccontextmanager
    async def lifespan(app):
        state['model'] = load_model(deploy_cfg)
        yield

    app = FastAPI(title='LingBot-VA-Suture inference server',
                  lifespan=lifespan)

    def frames_to_obs(frames):
        cam_keys = state['model'].config.obs_cam_keys
        if len(frames) != len(cam_keys):
            raise HTTPException(
                400, f'Expected {len(cam_keys)} images, got {len(frames)}.')
        return {key: decode_image(f) for key, f in zip(cam_keys, frames)}

    @app.get('/health')
    async def health():
        config = state['model'].config
        return {
            'status': 'ok',
            'cameras': config.obs_cam_keys,
            'num_cameras': len(config.obs_cam_keys),
            'task_prompts': config.task_prompts,
            'gpu': torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else 'N/A',
        }

    @app.post('/reset')
    async def reset(request: ResetRequest):
        model = state['model']
        model.reset(request.prompt)
        response = {'status': 'ok', 'prompt': request.prompt}
        if request.prompt not in model.config.task_prompts:
            response['warning'] = (
                'The model was not trained on this prompt. Trained prompts: '
                f'{model.config.task_prompts}')
            print(f"[/reset] WARNING: {response['warning']}")
        return response

    @app.post('/infer_chunk')
    async def infer_chunk(request: ChunkRequest):
        model = state['model']
        t0 = time.time()
        obs = frames_to_obs(request.frames) if request.frames else None
        try:
            actions, latents = model.infer_chunk(
                obs, seed=None if seed is None else seed + model.frame_st_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {
            'actions': actions.tolist(),
            'metadata': {
                'inference_time_sec': round(time.time() - t0, 2),
                'frame_st_id': model.frame_st_id,
                'num_action_steps': len(actions),
                'num_frames_to_cache_next': latents.shape[2] * 4,
            },
        }

    @app.post('/cache')
    async def cache(request: CacheRequest):
        model = state['model']
        t0 = time.time()
        obs = [frames_to_obs(frames) for frames in request.frames]
        try:
            num_padded = model.update_cache(obs)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if num_padded:
            print(f'[/cache] WARNING: padded {num_padded} frame(s) by '
                  f'repeating the last one ({len(obs)} received).')
        return {
            'status': 'ok',
            'frame_st_id': model.frame_st_id,
            'cache_time_sec': round(time.time() - t0, 2),
            'frames_received': len(obs),
            'frames_padded': num_padded,
        }

    @app.post('/infer')
    async def infer(request: InferRequest):
        model = state['model']
        t0 = time.time()
        actions, latents = model.generate(frames_to_obs(request.frames),
                                          request.text,
                                          request.num_chunks,
                                          seed=seed)
        video = model.decode_video(latents)
        return {
            'video_frames': [encode_image(frame) for frame in video],
            'actions': actions.tolist(),
            'metadata': {
                'inference_time_sec': round(time.time() - t0, 2),
                'num_chunks': request.num_chunks,
                'num_action_steps': len(actions),
                'num_generated_frames': len(video),
                'frame_size': list(video.shape[1:3]),
            },
        }

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config',
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            'deploy_config.yaml'))
    args = parser.parse_args()
    with open(args.config) as f:
        deploy_cfg = yaml.safe_load(f)

    # After this, the selected GPU is the only visible one: cuda:0
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',
                          str(deploy_cfg.get('gpu_id', 0)))
    uvicorn.run(create_app(deploy_cfg),
                host=deploy_cfg.get('host', '127.0.0.1'),
                port=deploy_cfg.get('port', 8100))


if __name__ == '__main__':
    main()
