# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified for LingBot-VA-Suture.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_suturebot_cfg = EasyDict(__name__='Config: VA suturebot')
va_suturebot_cfg.update(va_shared_cfg)

# lingbot-va-base: provides vae/, tokenizer/, text_encoder/ and the
# transformer/ that post-training starts from
va_suturebot_cfg.wan22_pretrained_model_name_or_path = '/path/to/lingbot-va-base'
# LingBot-VA-Suture weights: a directory that contains transformer/
va_suturebot_cfg.checkpoint_path = '/path/to/lingbot-va-suture'

va_suturebot_cfg.attn_window = 30
va_suturebot_cfg.frame_chunk_size = 4  # latent frames per chunk

va_suturebot_cfg.height = 480
va_suturebot_cfg.width = 640
va_suturebot_cfg.action_dim = 30
# 30 Hz actions, 10 fps video, 4 video frames per latent frame
va_suturebot_cfg.action_per_frame = 12

va_suturebot_cfg.obs_cam_keys = [
    'observation.images.cam_high',  # endoscope, left eye
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
]

# Prompts the model was trained on; the first one is the default.
# The released weights were trained with a single prompt for all sub-tasks.
va_suturebot_cfg.task_prompts = [
    'Perform surgical manipulation (hf_suturebot_separated).',
]

va_suturebot_cfg.guidance_scale = 5
va_suturebot_cfg.action_guidance_scale = 1

va_suturebot_cfg.num_inference_steps = 25
va_suturebot_cfg.video_exec_step = -1
va_suturebot_cfg.action_num_inference_steps = 50

va_suturebot_cfg.snr_shift = 5.0
va_suturebot_cfg.action_snr_shift = 1.0

# The model works on a 30-channel action layout:
#   [left pose (7), left joints (7), right pose (7), right joints (7),
#    left jaw (1), right jaw (1)]
# A pose is xyz + quaternion (xyzw). On the dVRK the left arm is PSM2 and the
# right arm is PSM1. The joint channels are not used.
va_suturebot_cfg.used_action_channel_ids = (
    list(range(0, 7))  # left pose (PSM2)
    + [28]  # left jaw (PSM2)
    + list(range(14, 21))  # right pose (PSM1)
    + [29]  # right jaw (PSM1)
)

# Channel 30 is a zero column that the unused joint channels are mapped to
inverse_used_action_channel_ids = list(range(va_suturebot_cfg.action_dim))
for dim in list(range(7, 14)) + list(range(21, 28)):
    inverse_used_action_channel_ids[dim] = va_suturebot_cfg.action_dim
va_suturebot_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# 1st / 99th percentiles of the relative actions of the training data
# (preprocess/compute_action_stats.py). They belong to the released weights.
va_suturebot_cfg.action_norm_method = 'quantiles'
va_suturebot_cfg.norm_stat = {
    "q01": [
        -0.009242, -0.005860, -0.009843, -0.257930, -0.238199, -0.263354, -0.995259,  # left pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left joints (unused)
        -0.020594, -0.013740, -0.019072, -0.281972, -0.251734, -0.353548, 0.805391,  # right pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right joints (unused)
        -0.349066,  # left jaw
        -0.349066,  # right jaw
    ],
    "q99": [
        0.012202, 0.005412, 0.011854, 0.265194, 0.207093, 0.290885, 1.000000,  # left pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left joints (unused)
        0.008644, 0.012457, 0.012125, 0.335563, 0.225452, 0.525455, 1.000000,  # right pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right joints (unused)
        0.926090,  # left jaw
        0.959298,  # right jaw
    ],
}
