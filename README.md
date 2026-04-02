# ⚡ PyTorch Project Template
* This repository is a codebase template for projects using PyTorch.
* It includes an example project that performs multi-class classification on the MNIST dataset using ConvNet and ConvNet2(with residual connections).

# 🚀 Version 2.1.0 Notes
- [x] Shell script-based execution
- [x] Removed logger (redirected stdout/stderr to log directory)
- [x] Added WandB logging
- [x] Unified <code>yaml</code> config (train/test)
- [x] Checkpoint-based saving (<code>.ckpt</code>)
- [x] Save test results as JSON
- [x] Added validation function (<code>engine/validate.py</code>)
- [x] Resume training from intermediate checkpoints
- [x] Subtasks execution
- [x] <code>DistributedDataParallel</code> training

### ✅ Planned for future release
- If you have idea, let me know! 

# 📂 Structure
```
├── README.md
├── configs
│   ├── convnet2_mnist_cls.yaml
│   └── convnet_mnist_cls.yaml
├── data
│   └── MNIST
├── datasets
│   ├── __init__.py
│   └── mnist_dataset.py
├── logs
│   ├── convnet2_mnist_cls
│   └── convnet_mnist_cls
├── models
│   ├── ConvNet
│   ├── ConvNet2
│   └── __init__.py
├── requirements.txt
├── scripts
│   ├── test_convnet2_mnist_cls.sh
│   ├── test_convnet_mnist_cls.sh
│   ├── train_convnet_mnist_cls.sh
│   ├── train_convnet_mnist_cls_ddp.sh
│   └── train_convnet2_mnist_cls.sh
├── subtasks
│   ├── 01
│   └── README.md
├── test.py
├── train.py
└── utils
    ├── __init__.py
    ├── engine.py
    ├── distributed.py
    ├── metrics.py
    └── save_ckpt.py
```

# 🔥 Installation, Environment Setup
```bash
git clone https://github.com/drawcodeboy/PyTorch-Project-Template.git
conda create -n template python=3.12
conda activate template
pip install -r requirements.txt
```
# 🔥 Execution
```bash
# Train (Check options wandb or resume)
bash scripts/train_convnet_mnist_cls.sh
bash scripts/train_convnet2_mnist_cls.sh

# Test
bash scripts/test_convnet_mnist_cls.sh
bash scripts/test_convnet2_mnist_cls.sh

# Subtask (Configure task via SUBTASK variable in shell script)
bash scripts/subtask.sh
```