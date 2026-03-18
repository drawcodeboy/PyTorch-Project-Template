#!/bin/bash

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate template

cd ~/torch_template
LOG_DIR=./logs/convnet2_mnist_cls
mkdir -p $LOG_DIR

OUT=$LOG_DIR/$(date +%Y%m%d-%H%M%S).out
ERR=$LOG_DIR/$(date +%Y%m%d-%H%M%S).err
exec 1> "$OUT"
exec 2> "$ERR"

export PYTHON_UNBUFFERED=1
python -u train.py --config=convnet2_mnist_cls