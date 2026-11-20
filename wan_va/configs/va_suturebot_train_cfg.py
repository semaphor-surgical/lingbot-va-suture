# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified for LingBot-VA-Suture.
import os

from easydict import EasyDict

from .va_suturebot_cfg import va_suturebot_cfg

va_suturebot_train_cfg = EasyDict(__name__='Config: VA suturebot train')
va_suturebot_train_cfg.update(va_suturebot_cfg)

# Dataset built with the scripts in preprocess/
va_suturebot_train_cfg.dataset_path = '/path/to/suturebot_dataset'
va_suturebot_train_cfg.empty_emb_path = os.path.join(
    va_suturebot_train_cfg.dataset_path, 'empty_emb.pt')

# None: start from the transformer of lingbot-va-base.
# A directory that contains transformer/: start from those weights instead.
va_suturebot_train_cfg.resume_from = None

va_suturebot_train_cfg.save_root = './train_out'
va_suturebot_train_cfg.save_interval = 50
va_suturebot_train_cfg.gc_interval = 50
va_suturebot_train_cfg.enable_wandb = False
va_suturebot_train_cfg.load_worker = 16

# Probability of replacing the prompt with the empty prompt
va_suturebot_train_cfg.cfg_prob = 0.1

va_suturebot_train_cfg.learning_rate = 1e-5
va_suturebot_train_cfg.beta1 = 0.9
va_suturebot_train_cfg.beta2 = 0.95
va_suturebot_train_cfg.weight_decay = 0.1
va_suturebot_train_cfg.warmup_steps = 100
va_suturebot_train_cfg.batch_size = 1  # per GPU
va_suturebot_train_cfg.gradient_accumulation_steps = 16
va_suturebot_train_cfg.num_steps = 100000
