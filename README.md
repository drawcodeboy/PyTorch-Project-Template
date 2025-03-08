# PyTorch Project Template
* This repository is a codebase template for projects using PyTorch.
* It includes an example project that performs multi-class classification on the MNIST dataset using ConvNet and ConvNet2(with residual connections).

# Installation & Environment Setup
```
# Docker Container Setup
docker pull ubuntu:22.04
docker run -itd --gpus=all --shm-size=16G --name=<container_name> ubuntu:22.04

# Ubuntu
apt-get update
apt-get install sudo
sudo apt-get install python3
sudo apt-get git
git clone https://github.com/drawcodeboy/PyTorch-Project-Template.git
cd <project_folder_name>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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