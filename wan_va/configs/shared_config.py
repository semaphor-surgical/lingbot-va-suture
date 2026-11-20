# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified for LingBot-VA-Suture.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.patch_size = (1, 2, 2)

# Keep the VAE and the text encoder on the CPU between uses to save GPU memory
va_shared_cfg.enable_offload = True
