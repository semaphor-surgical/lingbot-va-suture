# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for LingBot-VA-Suture.
import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    # FSDP is only used with more than one process
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if world_size > 1 else "gloo",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size)
