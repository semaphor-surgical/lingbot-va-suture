#!/usr/bin/env python3
"""Test client for the LingBot-VA-Suture inference server.

    # closed loop: /reset -> /infer_chunk -> /cache -> /infer_chunk ...
    python inference/va_deploy/test_client.py --mode server \
        --images endoscope.png wrist_left.png wrist_right.png

    # open loop: predicted video and actions from one observation
    python inference/va_deploy/test_client.py --mode i2va \
        --images endoscope.png wrist_left.png wrist_right.png

Without --images, blank frames are sent: enough to check that the server runs.
In closed-loop mode a real client sends the frames it observed while executing
each chunk; this script sends the first observation again.
"""
import argparse
import base64
import io
import json
import os

import numpy as np
import requests
from PIL import Image


def image_to_base64(image):
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def post(server, endpoint, payload):
    response = requests.post(f'{server}/{endpoint}', json=payload, timeout=1800)
    if not response.ok:
        raise SystemExit(f'/{endpoint} failed: {response.text[:500]}')
    return response.json()


def run_server_mode(args, frames):
    reset = post(args.server, 'reset', {'prompt': args.prompt})
    if 'warning' in reset:
        print(f"WARNING: {reset['warning']}")

    actions = []
    num_frames_to_cache = None
    for chunk_id in range(args.num_chunks):
        if chunk_id > 0:
            result = post(args.server, 'cache',
                          {'frames': [frames] * num_frames_to_cache})
            print(f"  /cache: {result['frames_received']} frames, "
                  f"{result['cache_time_sec']}s")
        result = post(args.server, 'infer_chunk',
                      {'frames': frames} if chunk_id == 0 else {})
        meta = result['metadata']
        num_frames_to_cache = meta['num_frames_to_cache_next']
        actions += result['actions']
        print(f"chunk {chunk_id}: {meta['num_action_steps']} actions, "
              f"{meta['inference_time_sec']}s")
    return actions


def run_i2va_mode(args, frames):
    result = post(args.server, 'infer', {
        'frames': frames,
        'text': args.prompt,
        'num_chunks': args.num_chunks
    })
    meta = result['metadata']
    for i, frame in enumerate(result['video_frames']):
        with open(os.path.join(args.output_dir, f'frame_{i:04d}.png'),
                  'wb') as f:
            f.write(base64.b64decode(frame))
    print(f"{meta['num_generated_frames']} predicted frames "
          f"({meta['frame_size'][0]}x{meta['frame_size'][1]}, the cameras "
          f"side by side), {meta['inference_time_sec']}s")
    return result['actions']


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--server', default='http://localhost:8100')
    parser.add_argument('--mode', choices=['server', 'i2va'], default='server')
    parser.add_argument('--images', nargs='*',
                        help='One image per camera, in the order of /health.')
    parser.add_argument('--prompt', default='needle pickup')
    parser.add_argument('--num_chunks', type=int, default=3)
    parser.add_argument('--output_dir', default='./va_test_output')
    args = parser.parse_args()

    health = requests.get(f'{args.server}/health', timeout=10).json()
    print(f"Server: {health['gpu']}, cameras: {health['cameras']}")
    print(f"Trained prompts: {health['task_prompts']}")

    if args.images:
        if len(args.images) != health['num_cameras']:
            raise SystemExit(f"The server expects {health['num_cameras']} images.")
        images = [Image.open(path) for path in args.images]
    else:
        images = [Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
                  ] * health['num_cameras']
    frames = [image_to_base64(image) for image in images]

    os.makedirs(args.output_dir, exist_ok=True)
    run = run_server_mode if args.mode == 'server' else run_i2va_mode
    actions = np.array(run(args, frames))

    with open(os.path.join(args.output_dir, 'actions.json'), 'w') as f:
        json.dump(actions.tolist(), f)
    print(f'Actions {actions.shape} saved to {args.output_dir}/actions.json')


if __name__ == '__main__':
    main()
