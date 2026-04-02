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

# Distributed Training Options:
#   One node, multiple processes (e.g., 2 processes for 2 GPUs)
#   torchrun --nnodes=1 --nproc_per_node=2 train.py

#   --distributed (enable distributed training)
#   --backend (distributed backend, e.g., gloo, nccl; default: nccl)
#       if you use CPU, use gloo. If you use GPU, use nccl for better performance.
#       So, please check config.yaml for the backend setting.
#   --init_method (initialization method for distributed training, default: env://)
#       For single-node multi-process training, env:// is sufficient.
#       For multi-node training, you may need to specify a shared file system or use TCP initialization.

export PYTHON_UNBUFFERED=1
torchrun --nnodes=1 --nproc_per_node=2 train.py \
--config=convnet_mnist_cls --distributed --backend=gloo