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

# Sub-task prompts the model was trained on
va_suturebot_cfg.task_prompts = [
    'needle pickup',
    'needle handover',
    'needle throw',
    'needle extract',
    'knot tying',
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
        -0.009022, -0.005911, -0.007823, -0.184765, -0.239764, -0.264861, 0.923451,  # left pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left joints (unused)
        -0.020887, -0.012911, -0.018214, -0.224219, -0.242405, -0.343701, 0.798794,  # right pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right joints (unused)
        -0.349066,  # left jaw
        -0.349066,  # right jaw
    ],
    "q99": [
        0.012338, 0.005136, 0.012233, 0.288708, 0.159472, 0.160411, 1.000000,  # left pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left joints (unused)
        0.008082, 0.013642, 0.012386, 0.340548, 0.209883, 0.534306, 1.000000,  # right pose
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right joints (unused)
        0.919616,  # left jaw
        0.924350,  # right jaw
    ],
}
