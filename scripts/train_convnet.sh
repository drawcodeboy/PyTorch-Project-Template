#!/bin/bash

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate template

cd ~/torch_template
mkdir -p ./logs/convnet

OUT=./logs/convnet/$(date +%Y%m%d-%H%M%S).out
ERR=./logs/convnet/$(date +%Y%m%d-%H%M%S).err
exec 1> "$OUT"
exec 2> "$ERR"

export PYTHON_UNBUFFERED=1
python -u train.py --config=convnet