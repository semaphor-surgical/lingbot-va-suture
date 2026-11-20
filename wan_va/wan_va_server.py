# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified for LingBot-VA-Suture.
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from tqdm import tqdm

from .modules import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from .utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, logger

VAE_TEMPORAL_STRIDE = 4  # video frames per latent frame


class VA_Server:
    """LingBot-VA-Suture policy.

    The model generates the future in chunks of `frame_chunk_size` latent video
    frames. For each chunk it first denoises the video latents, then the actions
    that realise them. Past chunks are kept as transformer keys / values (the
    KV cache), so the context is never recomputed.

    Closed loop (one session):
        reset(prompt)
        actions = infer_chunk(first_frames)
        repeat:  execute actions -> update_cache(observed_frames)
                 actions = infer_chunk()

    Open loop:
        actions, latents = generate(first_frames, prompt, num_chunks)
    """

    def __init__(self, config):
        self.config = config
        self.dtype = config.param_dtype
        self.device = torch.device(getattr(config, 'device', 'cuda:0'))
        self.enable_offload = config.enable_offload
        self.cache_name = 'pos'

        self.scheduler = FlowMatchScheduler(shift=config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)

        base = config.wan22_pretrained_model_name_or_path
        offload_device = 'cpu' if self.enable_offload else self.device
        self.vae = load_vae(os.path.join(base, 'vae'),
                            torch_dtype=self.dtype,
                            torch_device=offload_device)
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self.tokenizer = load_tokenizer(os.path.join(base, 'tokenizer'))
        self.text_encoder = load_text_encoder(os.path.join(
            base, 'text_encoder'),
                                              torch_dtype=self.dtype,
                                              torch_device=offload_device)
        self.transformer = load_transformer(os.path.join(
            config.checkpoint_path, 'transformer'),
                                            torch_dtype=self.dtype,
                                            torch_device='cpu',
                                            attn_mode='torch')
        self.transformer.eval().requires_grad_(False)
        self.transformer.to(self.dtype).to(self.device)

        self.height, self.width = config.height, config.width
        self.action_per_frame = config.action_per_frame
        self.latent_height = self.height // 16
        self.latent_width = self.width // 16 * len(config.obs_cam_keys)

        self.action_mask = torch.zeros([config.action_dim]).bool()
        self.action_mask[config.used_action_channel_ids] = True
        self.actions_q01 = torch.tensor(config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)

        self.use_cfg = config.guidance_scale > 1 or config.action_guidance_scale > 1
        self._clear_session()

    # ------------------------------------------------------------------ #
    # Text, video and action encoding
    # ------------------------------------------------------------------ #
    def _encode_text(self, text, max_sequence_length=512):
        text_inputs = self.tokenizer(
            [prompt_clean(text)],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_len = int(mask.gt(0).sum())

        text_encoder_device = next(self.text_encoder.parameters()).device
        embeds = self.text_encoder(input_ids.to(text_encoder_device),
                                   mask.to(text_encoder_device)).last_hidden_state
        embeds = embeds.to(dtype=self.dtype, device=self.device)
        # zero the padded positions
        embeds[:, seq_len:] = 0
        return embeds

    def _vae_to_device(self):
        """Move an offloaded VAE (and its streaming cache) to the GPU."""
        self.vae.to(self.device)
        self.streaming_vae.feat_cache = [
            c.to(self.device) if torch.is_tensor(c) else c
            for c in self.streaming_vae.feat_cache
        ]

    def _vae_offload(self):
        if self.enable_offload:
            self.vae.to('cpu')
            torch.cuda.empty_cache()

    def _frames_to_video(self, frames):
        """frames: list over time of {camera key: HxWx3 uint8 array}.

        Returns a [num_cameras, 3, T, H, W] tensor in [-1, 1].
        """
        videos = []
        for key in self.config.obs_cam_keys:
            video = torch.from_numpy(np.stack([f[key] for f in frames]))
            video = video.float().permute(3, 0, 1, 2)
            video = F.interpolate(video,
                                  size=(self.height, self.width),
                                  mode='bilinear',
                                  align_corners=False)
            videos.append(video.unsqueeze(0))
        return torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0

    def _encode_video(self, video):
        """Encode the next frames of the observation stream.

        The VAE is causal in time; its convolution cache is kept between calls,
        so consecutive calls encode one continuous video (as in training).
        Returns [1, C, T_latent, H, W * num_cameras]: the cameras side by side.
        """
        self._vae_to_device()
        enc_out = self.streaming_vae.encode_chunk(
            video.to(self.device).to(self.dtype))
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        mu = ((mu.float() - mean.to(mu.device)) *
              (1.0 / std).to(mu.device)).to(mu)
        self._vae_offload()
        return torch.cat(mu.split(1, dim=0), dim=-1)

    def _normalize_actions(self, actions):
        """[16, F, N] actions -> [1, 30, F, N, 1] model input."""
        full = torch.zeros(self.config.action_dim, *actions.shape[1:])
        full[self.config.used_action_channel_ids] = torch.from_numpy(
            actions).float()
        full = (full - self.actions_q01) / (
            self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        return full.unsqueeze(0).unsqueeze(-1)

    def _denormalize_actions(self, actions):
        """[1, 30, F, N, 1] model output -> [16, F, N] actions."""
        actions = actions.cpu()[0, ..., 0]
        actions = (actions + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                       1e-6) + self.actions_q01
        actions = actions.numpy()[self.config.used_action_channel_ids]
        if not np.isfinite(actions).all():
            raise RuntimeError('The model produced non-finite actions.')

        # Never return values outside the range seen in training
        # (+-1.5 in normalized units, the bound used for the training targets)
        used = self.config.used_action_channel_ids
        q01 = self.actions_q01.numpy()[used]
        q99 = self.actions_q99.numpy()[used]
        margin = 0.25 * (q99 - q01)
        clipped = np.clip(actions, q01 - margin, q99 + margin)
        num_clipped = int((clipped != actions).sum())
        if num_clipped:
            logger.warning(
                f'{num_clipped} action values were clipped to the training range'
            )
        return clipped

    # ------------------------------------------------------------------ #
    # Transformer inputs
    # ------------------------------------------------------------------ #
    def _latent_input(self, latents, t=0, cond=None):
        timesteps = torch.ones([latents.shape[2]],
                               dtype=torch.float32,
                               device=self.device) * t
        patch_size = self.config.patch_size
        grid_id = get_mesh_id(latents.shape[-3] // patch_size[0],
                              latents.shape[-2] // patch_size[1],
                              latents.shape[-1] // patch_size[2], 0, 1,
                              self.frame_st_id).to(self.device)
        if cond is not None:  # the first frame is given
            latents[:, :, 0:1] = cond
            timesteps[0:1] *= 0
        return self._batch_input(latents, timesteps, grid_id)

    def _action_input(self, actions, t=0, cond=None):
        timesteps = torch.ones([actions.shape[2]],
                               dtype=torch.float32,
                               device=self.device) * t
        grid_id = get_mesh_id(actions.shape[-3],
                              actions.shape[-2],
                              actions.shape[-1],
                              1,
                              1,
                              self.frame_st_id,
                              action=True).to(self.device)
        if cond is not None:
            actions[:, :, 0:1] = cond
            timesteps[0:1] *= 0
        actions[:, ~self.action_mask] *= 0
        return self._batch_input(actions, timesteps, grid_id)

    def _batch_input(self, noisy_latents, timesteps, grid_id):
        """Add the batch axis; with guidance the batch is [prompt, no prompt]."""
        if self.use_cfg:
            return dict(
                noisy_latents=noisy_latents.repeat(2, 1, 1, 1, 1),
                text_emb=torch.cat(
                    [self.prompt_embeds, self.negative_prompt_embeds], dim=0),
                grid_id=grid_id[None].repeat(2, 1, 1),
                timesteps=timesteps[None].repeat(2, 1),
            )
        return dict(
            noisy_latents=noisy_latents,
            text_emb=self.prompt_embeds.clone(),
            grid_id=grid_id[None],
            timesteps=timesteps[None],
        )

    @staticmethod
    def _guide(noise_pred, guidance_scale):
        if guidance_scale > 1:
            return noise_pred[1:] + guidance_scale * (noise_pred[:1] -
                                                      noise_pred[1:])
        return noise_pred[:1]

    # ------------------------------------------------------------------ #
    # Session
    # ------------------------------------------------------------------ #
    def _clear_session(self):
        self.frame_st_id = 0
        self.init_latent = None
        self.last_actions = None
        self.prompt_embeds = self.negative_prompt_embeds = None
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

    @torch.no_grad()
    def reset(self, prompt):
        """Start a new session: clear all caches and set the task prompt."""
        config = self.config
        self._clear_session()

        patch_size = config.patch_size
        latent_token_per_chunk = (
            config.frame_chunk_size * self.latent_height *
            self.latent_width) // (patch_size[0] * patch_size[1] *
                                   patch_size[2])
        action_token_per_chunk = config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size=2 if self.use_cfg else 1)

        self.prompt_embeds = self._encode_text(prompt)
        self.negative_prompt_embeds = self._encode_text(
            '') if self.use_cfg else None
        torch.cuda.empty_cache()

    def _denoise_chunk(self):
        """Generate the chunk that starts at latent frame `self.frame_st_id`.

        Returns the normalized actions [1, 30, F, N, 1] and the video latents
        [1, C, F, H, W]. The chunk is left in the KV cache as a prediction.
        """
        config = self.config
        first_chunk = self.frame_st_id == 0
        frame_chunk_size = config.frame_chunk_size

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        self.scheduler.set_timesteps(config.num_inference_steps)
        self.action_scheduler.set_timesteps(config.action_num_inference_steps)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), value=0)
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1),
                                 value=0)
        video_exec_step = config.video_exec_step
        if video_exec_step != -1:  # stop the video denoising early
            timesteps = timesteps[:video_exec_step]

        # 1. Video: the first frame of a session is the observed one
        latent_cond = self.init_latent[:, :, 0:1].to(
            self.dtype) if first_chunk else None
        for i, t in enumerate(tqdm(timesteps)):
            last_step = i == len(timesteps) - 1
            noise_pred = self.transformer(
                self._latent_input(latents, t, latent_cond),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False)
            if not last_step or video_exec_step != -1:
                noise_pred = data_seq_to_patch(
                    config.patch_size,
                    noise_pred,
                    frame_chunk_size,
                    self.latent_height,
                    self.latent_width,
                    batch_size=2 if self.use_cfg else 1)
                noise_pred = self._guide(noise_pred, config.guidance_scale)
                latents = self.scheduler.step(noise_pred, t, latents)
            if first_chunk:
                latents[:, :, 0:1] = latent_cond

        # 2. Actions: the first action frame of a session is an empty look-back
        action_cond = torch.zeros(
            [1, config.action_dim, 1, self.action_per_frame, 1],
            device=self.device,
            dtype=self.dtype) if first_chunk else None
        for i, t in enumerate(tqdm(action_timesteps)):
            last_step = i == len(action_timesteps) - 1
            noise_pred = self.transformer(
                self._action_input(actions, t, action_cond),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True)
            if not last_step:
                noise_pred = rearrange(noise_pred,
                                       'b (f n) c -> b c f n 1',
                                       f=frame_chunk_size)
                noise_pred = self._guide(noise_pred,
                                         config.action_guidance_scale)
                actions = self.action_scheduler.step(noise_pred, t, actions)
            if first_chunk:
                actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0
        torch.cuda.empty_cache()
        return actions, latents

    @torch.no_grad()
    def infer_chunk(self, frames=None, seed=None):
        """Predict the next chunk of actions.

        frames: {camera key: HxWx3 uint8 array}, the current observation.
            Required for the first chunk of a session, ignored afterwards.
        Returns actions [frame_chunk_size * action_per_frame, 16] and the
        predicted video latents.
        """
        if self.prompt_embeds is None:
            raise ValueError('Call reset before infer_chunk.')
        if self.frame_st_id == 0 and self.init_latent is None:
            if frames is None:
                raise ValueError(
                    'The first chunk of a session needs the current frames.')
            self.init_latent = self._encode_video(
                self._frames_to_video([frames]))

        # drop the prediction of an earlier call for the same chunk
        self.transformer.clear_pred_cache(self.cache_name)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        actions, latents = self._denoise_chunk()
        self.last_actions = self._denormalize_actions(actions)
        return self.last_actions.reshape(len(self.last_actions), -1).T, latents

    @torch.no_grad()
    def update_cache(self, frames):
        """Replace the predicted chunk in the KV cache by what really happened.

        frames: list over time of {camera key: HxWx3 uint8 array}: the 10 fps
            frames observed while the last predicted chunk was executed, one
            latent frame for every 4 of them. A count that is not a multiple
            of 4 is padded by repeating the last frame.
        Returns the number of padded frames.
        """
        if self.last_actions is None:
            raise ValueError('Call infer_chunk before update_cache.')
        num_padded = -len(frames) % VAE_TEMPORAL_STRIDE
        num_latent_frames = (len(frames) + num_padded) // VAE_TEMPORAL_STRIDE
        if num_latent_frames != self.last_actions.shape[1]:
            raise ValueError(
                f'{len(frames)} frames give {num_latent_frames} latent frames, '
                f'but the executed chunk has {self.last_actions.shape[1]}. '
                f'Send {self.config.frame_chunk_size * VAE_TEMPORAL_STRIDE} frames.'
            )

        video = self._frames_to_video(frames)
        if num_padded:
            video = torch.cat(
                [video, video[:, :, -1:].repeat(1, 1, num_padded, 1, 1)],
                dim=2)
        latents = self._encode_video(video)
        actions = self._normalize_actions(self.last_actions).to(latents)

        self.transformer.clear_pred_cache(self.cache_name)
        self.transformer(self._latent_input(latents),
                         update_cache=2,
                         cache_name=self.cache_name,
                         action_mode=False)
        self.transformer(self._action_input(actions),
                         update_cache=2,
                         cache_name=self.cache_name,
                         action_mode=True)
        self.frame_st_id += latents.shape[2]
        torch.cuda.empty_cache()
        return num_padded

    @torch.no_grad()
    def generate(self, frames, prompt, num_chunks, seed=None):
        """Open-loop rollout from one observation.

        Every chunk is conditioned on the model's own previous predictions.
        Returns actions [num_chunks * frame_chunk_size * action_per_frame, 16]
        and the predicted video latents.
        """
        self.reset(prompt)
        self.init_latent = self._encode_video(self._frames_to_video([frames]))
        actions_lst, latents_lst = [], []
        for chunk_id in range(num_chunks):
            if seed is not None:
                torch.manual_seed(seed + chunk_id)
                torch.cuda.manual_seed(seed + chunk_id)
            self.frame_st_id = chunk_id * self.config.frame_chunk_size
            actions, latents = self._denoise_chunk()
            actions_lst.append(self._denormalize_actions(actions))
            latents_lst.append(latents)
        self._clear_session()

        actions = np.concatenate(actions_lst, axis=1)
        return actions.reshape(len(actions), -1).T, torch.cat(latents_lst,
                                                              dim=2)

    @torch.no_grad()
    def decode_video(self, latents):
        """Video latents -> uint8 frames [T, H, W * num_cameras, 3]."""
        self.vae.to(self.device)
        latents = latents.to(self.vae.dtype)
        mean = torch.tensor(self.vae.config.latents_mean).view(
            1, -1, 1, 1, 1).to(latents)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, -1, 1, 1, 1).to(latents)
        video = self.vae.decode(latents / inv_std + mean,
                                return_dict=False)[0]
        video = self.video_processor.postprocess_video(video,
                                                       output_type='np')[0]
        self._vae_offload()
        return (np.clip(video, 0, 1) * 255).astype(np.uint8)
