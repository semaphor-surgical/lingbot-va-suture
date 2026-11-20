#!/usr/bin/bash
# Post-train LingBot-VA on SutureBot with FSDP on one node.
#   NGPU=8 bash script/run_suturebot_posttrain.sh [--save-root DIR]
# Paths and hyper-parameters: wan_va/configs/va_suturebot_train_cfg.py
set -e
cd "$(dirname "$0")/.."

NGPU=${NGPU:-8}
MASTER_PORT=${MASTER_PORT:-29502}

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --master_port ${MASTER_PORT} \
    --local-ranks-filter=0 \
    --tee 3 \
    -m wan_va.train --config-name suturebot_train "$@"
