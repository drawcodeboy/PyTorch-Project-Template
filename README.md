# PyTorch Project Template
* 본 Repository는 PyTorch를 통해 프로젝트를 수행할 때 기반이 되는 코드베이스 템플릿입니다.

# Installation
```
# Docker Container Setup
docker pull ubuntu:22.04
docker run -itd --gpus=all --shm-size=16G --name=<container_name> <image_name>

# Ubuntu
apt-get update
apt-get install sudo
sudo apt-get install python3
python3 -m venv .venv
source .venv/bin/activate
```
# Train & Test
```
# Train
python train.py --config=convnet
python train.py --config=convnet2

# Test
python test.py --config=convnet
python test.py --config=convnet2
```
# Performances
```
# ConvNet (20 Epochs)
Accuracy: 0.9540
F1-Score(Macro): 0.9539
Precision(Macro): 0.9572
Recall(Macro): 0.9537

# ConvNet2 (20 Epochs)
Accuracy: 0.9700
F1-Score(Macro): 0.9699
Precision(Macro): 0.9701
Recall(Macro): 0.9701
```