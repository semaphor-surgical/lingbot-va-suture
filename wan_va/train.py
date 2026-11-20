# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified for LingBot-VA-Suture.
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader, DistributedSampler

from .configs import VA_CONFIGS
from .dataset import LatentLeRobotDataset
from .distributed.fsdp import apply_ac, shard_model
from .distributed.util import _configure_model, init_distributed
from .modules import load_transformer
from .utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    sample_timestep_id,
    warmup_constant_lambda,
)


class Trainer:

    def __init__(self, config):
        self.config = config
        self.step = 0
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.save_dir = Path(config.save_root)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.wandb = None
        if config.enable_wandb and config.rank == 0:
            import wandb
            wandb.init(project=os.getenv("WANDB_PROJECT", "lingbot-va-suture"),
                       config=dict(config))
            self.wandb = wandb

        # Transformer: FSDP over the GPUs, activation checkpointing per block
        model_dir = config.resume_from or config.wan22_pretrained_model_name_or_path
        logger.info(f"Loading transformer from {model_dir}")
        self.transformer = load_transformer(
            os.path.join(model_dir, 'transformer'),
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode='flex',
        )
        apply_ac(self.transformer)
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(
                step, warmup_steps=config.warmup_steps))

        train_dataset = LatentLeRobotDataset(config)
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=config.load_worker,
            sampler=self.train_sampler,
            drop_last=True,
            pin_memory=True,
        )

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

    def _batches(self):
        """Endless stream of batches."""
        epoch = 0
        while True:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for batch in self.train_loader:
                yield {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                }
            epoch += 1

    @torch.no_grad()
    def _add_noise(self,
                   latent,
                   train_scheduler,
                   action_mask=None,
                   action_mode=False,
                   noisy_cond_prob=0.):
        """Noise every latent frame with its own timestep.

        Returns the noisy sequence (the denoising targets) together with the
        clean sequence, which the model sees as the history of each frame.
        """
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(
            batch_size=F,
            num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(
            device=self.device)
        noisy_latents = train_scheduler.add_noise(latent,
                                                  noise,
                                                  timesteps,
                                                  t_dim=2,
                                                  timestep_id=timestep_ids)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = (1, 1, 1) if action_mode else self.patch_size
        grid_id = get_mesh_id(latent.shape[-3] // patch_f,
                              latent.shape[-2] // patch_h,
                              latent.shape[-1] // patch_w,
                              t=1 if action_mode else 0,
                              f_w=1,
                              f_shift=0,
                              action=action_mode).to(self.device)
        grid_id = grid_id[None].repeat(B, 1, 1)

        # Sometimes the history is noisy too, as predicted frames are at inference
        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(
                device=self.device)
            latent = train_scheduler.add_noise(latent,
                                               noise,
                                               cond_timesteps,
                                               t_dim=2,
                                               timestep_id=cond_timestep_ids)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            timestep_ids=timestep_ids[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch):
        latent_dict = self._add_noise(
            latent=batch['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mode=False,
            noisy_cond_prob=0.5)
        action_dict = self._add_noise(
            latent=batch['actions'],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch['actions_mask'],
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch['text_emb']
        action_dict['text_emb'] = batch['text_emb']
        action_dict['actions_mask'] = batch['actions_mask']

        # The chunk size and the attention window vary during training
        return {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1, )).item(),
            'window_size': torch.randint(4, 65, (1, )).item(),
        }

    def compute_loss(self, input_dict, pred):
        """Flow-matching losses of the video latents and of the actions."""
        latent_dict, action_dict = input_dict['latent_dict'], input_dict[
            'action_dict']
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred,
                                'b (f n) c -> b c f n 1',
                                f=action_dict['targets'].shape[-3])
        latent_pred = data_seq_to_patch(self.patch_size,
                                        latent_pred,
                                        latent_dict['targets'].shape[-3],
                                        latent_dict['targets'].shape[-2],
                                        latent_dict['targets'].shape[-1],
                                        batch_size=latent_pred.shape[0])

        def frame_weights(scheduler, noise_dict):
            B, F = noise_dict['timesteps'].shape
            return scheduler.training_weight(
                noise_dict['timesteps'].flatten(),
                timestep_id=noise_dict['timestep_ids'].flatten()).reshape(B, F)

        def per_frame(x):  # (B, C, F, H, W) -> (B * F, H * W * C)
            return x.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)

        # Video: mean over the elements of a frame, then over the frames
        latent_loss = F.mse_loss(latent_pred.float(),
                                 latent_dict['targets'].float().detach(),
                                 reduction='none')
        latent_loss = latent_loss * frame_weights(
            self.train_scheduler_latent, latent_dict)[:, None, :, None, None]
        latent_loss = per_frame(latent_loss)
        latent_loss = (latent_loss.sum(dim=1) /
                       (latent_loss.shape[1] + 1e-6)).mean()

        # Actions: the same, over the supervised elements and frames only
        action_mask = action_dict['actions_mask'].float()
        action_loss = F.mse_loss(action_pred.float(),
                                 action_dict['targets'].float().detach(),
                                 reduction='none')
        action_loss = action_loss * frame_weights(
            self.train_scheduler_action, action_dict)[:, None, :, None, None]
        action_loss = per_frame(action_loss * action_mask).sum(dim=1)
        mask_per_frame = per_frame(action_mask).sum(dim=1)
        supervised = mask_per_frame > 0
        action_loss = (action_loss[supervised] /
                       mask_per_frame[supervised]).mean()

        return latent_loss, action_loss

    def _train_step(self, batches):
        """One optimizer step over `gradient_accumulation_steps` batches."""
        latent_loss_sum = action_loss_sum = 0.
        for i in range(self.gradient_accumulation_steps):
            input_dict = self._prepare_input_dict(next(batches))
            # FSDP: reduce the gradients across GPUs on the last batch only
            if hasattr(self.transformer, 'set_requires_gradient_sync'):
                self.transformer.set_requires_gradient_sync(
                    i == self.gradient_accumulation_steps - 1)

            output = self.transformer(input_dict, train_mode=True)
            latent_loss, action_loss = self.compute_loss(input_dict, output)
            loss = (latent_loss +
                    action_loss) / self.gradient_accumulation_steps
            loss.backward()
            latent_loss_sum += latent_loss.detach()
            action_loss_sum += action_loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.transformer.parameters(), 2.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        losses = torch.stack([latent_loss_sum, action_loss_sum
                              ]) / self.gradient_accumulation_steps
        if dist.get_world_size() > 1:
            dist.all_reduce(losses, op=dist.ReduceOp.AVG)
        return losses[0].item(), losses[1].item(), grad_norm.item()

    def save_checkpoint(self):
        """Save the transformer in the layout of the pretrained model."""
        state_dict = get_model_state_dict(
            self.transformer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if self.config.rank == 0:
            transformer_dir = self.save_dir / f"checkpoint_step_{self.step}" / "transformer"
            transformer_dir.mkdir(parents=True, exist_ok=True)
            save_file({k: v.to(torch.bfloat16)
                       for k, v in state_dict.items()},
                      transformer_dir / "diffusion_pytorch_model.safetensors")
            config_dict = dict(self.transformer.config)
            config_dict.pop('_name_or_path', None)
            with open(transformer_dir / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved {transformer_dir}")
        dist.barrier()

    def train(self):
        config = self.config
        logger.info(f"Training for {config.num_steps} steps")
        self.optimizer.zero_grad()
        batches = self._batches()

        while self.step < config.num_steps:
            latent_loss, action_loss, grad_norm = self._train_step(batches)
            lr = self.lr_scheduler.get_last_lr()[0]
            self.step += 1

            if config.rank == 0:
                if self.step % 10 == 0 or self.step == 1:
                    logger.info(f"step {self.step}/{config.num_steps} | "
                                f"latent_loss={latent_loss:.4f} | "
                                f"action_loss={action_loss:.4f} | "
                                f"grad_norm={grad_norm:.2f} | lr={lr:.2e}")
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            'latent_loss': latent_loss,
                            'action_loss': action_loss,
                            'grad_norm': grad_norm,
                            'lr': lr,
                        },
                        step=self.step)

            if self.step % config.gc_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()
            if self.step % config.save_interval == 0:
                self.save_checkpoint()

        logger.info("Training completed")


def main():
    parser = argparse.ArgumentParser(
        description="Post-train LingBot-VA on SutureBot")
    parser.add_argument("--config-name", type=str, default='suturebot_train')
    parser.add_argument("--save-root",
                        type=str,
                        default=None,
                        help="Directory for the checkpoints")
    args = parser.parse_args()

    config = VA_CONFIGS[args.config_name]
    config.rank = int(os.getenv("RANK", 0))
    config.local_rank = int(os.getenv("LOCAL_RANK", 0))
    config.world_size = int(os.getenv("WORLD_SIZE", 1))
    if args.save_root is not None:
        config.save_root = args.save_root

    init_distributed(config.world_size, config.local_rank, config.rank)
    Trainer(config).train()


if __name__ == "__main__":
    init_logger()
    main()
