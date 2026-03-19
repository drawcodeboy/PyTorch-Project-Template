#!/bin/bash

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate template

cd ~/torch_template
LOG_DIR=./logs/subtasks
mkdir -p $LOG_DIR
SUBTASK=01

OUT=$LOG_DIR/$SUBTASK-$(date +%Y%m%d-%H%M%S).out
ERR=$LOG_DIR/$SUBTASK-$(date +%Y%m%d-%H%M%S).err
exec 1> "$OUT"
exec 2> "$ERR"

export PYTHON_UNBUFFERED=1
python -u subtasks/$SUBTASK/exec.py