#!/bin/bash

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate template

cd ~/torch_template
LOG_DIR=./logs/convnet_mnist_cls
mkdir -p $LOG_DIR

OUT=$LOG_DIR/$(date +%Y%m%d-%H%M%S).out
ERR=$LOG_DIR/$(date +%Y%m%d-%H%M%S).err
exec 1> "$OUT"
exec 2> "$ERR"

# Options:
#   export WANDB_API_KEY=your_wandb_api_key_here
#   --use_wandb (log to wandb)
#   --resume (load last.ckpt)

export PYTHON_UNBUFFERED=1
python -u train.py --config=convnet_mnist_cls --resume