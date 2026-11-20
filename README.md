# LingBot-VA-Suture

LingBot-VA-Suture adapts [LingBot-VA](https://github.com/Robbyant/lingbot-va), a video-action world model, to suturing with the da Vinci Research Kit (dVRK). The released weights are `lingbot-va-base` post-trained on the SutureBot data of [Open-H-Embodiment](https://arxiv.org/abs/2604.21017).

At every step the model predicts a chunk of future video latents and the end-effector actions that realise them. Past chunks stay in a transformer KV cache, so a session runs as a stream.

This repository holds the code to run the model and to post-train it. It is a reduced and modified copy of the LingBot-VA code base.

## Model

| | |
|:--|:--|
| Cameras | endoscope (left eye), left wrist, right wrist; resized to 480 x 640 |
| Prompt | one sub-task: `needle pickup`, `needle handover`, `needle throw`, `needle extract`, `knot tying` |
| Actions | 16 values per step at 30 Hz: left arm (PSM2) pose and jaw, then right arm (PSM1) pose and jaw |
| Chunk | 48 actions (1.6 s) and 4 latent video frames (16 frames at 10 fps) |

A pose is a position (xyz) and a quaternion (xyzw). Poses are relative to the pose at the first observation of the session: the position is `p - p0` and the rotation is `R0^-1 R`. The jaw value is the absolute jaw angle in radians. Training windows are at most 10 s long.

In the first chunk of a session the first 12 actions are placeholders, not predictions: that slot belongs to the look-back frame of the model. Skip them.

## Setup

Python 3.10, PyTorch 2.9, one CUDA GPU for inference.

```bash
pip install -r requirements.txt
```

Download the base model, which provides the VAE, the tokenizer and the text encoder:

```bash
hf download robbyant/lingbot-va-base --local-dir ./lingbot-va-base
```

The LingBot-VA-Suture transformer is in a Hugging Face bucket. `hf buckets` needs a recent `huggingface_hub`, newer than `transformers` 4.55 accepts, so run it from a separate environment:

```bash
python -m venv hf_env && hf_env/bin/pip install -U huggingface_hub
hf_env/bin/hf buckets sync hf://buckets/semaphorsurgical/lingbot-va-suture ./lingbot-va-suture-weights
```

## Inference

```bash
cd inference/va_deploy
cp deploy_config.template.yaml deploy_config.yaml   # set base_model and checkpoint
python server.py --config deploy_config.yaml
```

Test it from a second terminal. Without `--images` the client sends blank frames, which is enough to check that the server runs.

```bash
# closed loop
python test_client.py --mode server --prompt "needle pickup" \
    --images endoscope_left.png wrist_left.png wrist_right.png
# open loop: predicted video frames and actions are saved to ./va_test_output
python test_client.py --mode i2va --prompt "needle pickup" \
    --images endoscope_left.png wrist_left.png wrist_right.png
```

Images are sent as base64-encoded PNGs, one per camera, in the order above.

| Endpoint | Request | Result |
|:--|:--|:--|
| `GET /health` | | camera order and trained prompts |
| `POST /reset` | `prompt` | starts a session |
| `POST /infer_chunk` | `frames`: current observation, first chunk only | `actions` [48, 16] |
| `POST /cache` | `frames`: the frames observed at 10 fps while the chunk was executed | updates the KV cache |
| `POST /infer` | `frames`, `text`, `num_chunks` | open-loop rollout: `actions` and decoded `video_frames` |

A closed-loop session is `/reset`, then `/infer_chunk` and `/cache` in turn. `/cache` takes 16 frames per chunk. With fewer frames, down to 13, the last one is repeated. The server keeps the actions it returned and writes them into the cache together with the observed frames.

`/infer` needs no robot: starting from one observation, every chunk continues from the model's own prediction.

## Post-training

Training runs on video latents that are computed once. Starting from SutureBot in the LeRobot v2 format of Open-H-Embodiment, with its episodes split by sub-task:

```bash
# 1. actions, camera names and segments in the layout of LingBot-VA
python preprocess/prepare_dataset.py --input_dir /path/to/suturebot --output_dir /path/to/suturebot_dataset
# 2. VAE latents of the videos and text embeddings of the prompts
python preprocess/extract_latents.py --dataset_path /path/to/suturebot_dataset --model_path ./lingbot-va-base
# 3. action normalization statistics
python preprocess/compute_action_stats.py --dataset_path /path/to/suturebot_dataset
```

Step 3 prints `norm_stat`. Put it into `wan_va/configs/va_suturebot_cfg.py`. The values in the repository belong to the released weights.

Set `dataset_path` and the model paths in `wan_va/configs/`, then start the training with FSDP on all GPUs of one node:

```bash
NGPU=8 bash script/run_suturebot_posttrain.sh --save-root ./train_out
```

Checkpoints are written as `checkpoint_step_N/transformer/`, the layout that `checkpoint` in the deploy config expects.

## Repository

```
wan_va/
  modules/model.py       transformer (shared video and action backbone)
  wan_va_server.py       inference: chunk generation, KV cache, action decoding
  train.py               post-training
  dataset/               training windows from latents and actions
  configs/               model, inference and training settings
inference/va_deploy/     HTTP server and test client
preprocess/              dataset conversion, latent extraction, action statistics
script/                  training launcher
```

## License and citation

The code is released under the Apache License 2.0, as is LingBot-VA, from which it is derived: see [LICENSE.txt](LICENSE.txt).

If you use this model, please cite LingBot-VA and Open-H-Embodiment:

```bibtex
@misc{lingbot-va,
  title        = {LingBot-VA: Causal World Modeling for Robot Control},
  author       = {{Robbyant}},
  year         = {2026},
  howpublished = {\url{https://github.com/Robbyant/lingbot-va}},
  note         = {arXiv:2601.21998}
}

@misc{open-h-embodiment,
  title         = {Open-H-Embodiment: A Large-Scale Dataset for Enabling Foundation Models in Medical Robotics},
  author        = {{Open-H-Embodiment Consortium}},
  year          = {2026},
  eprint        = {2604.21017},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
