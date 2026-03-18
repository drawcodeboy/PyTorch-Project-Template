# PyTorch Project Template
* This repository is a codebase template for projects using PyTorch.
* It includes an example project that performs multi-class classification on the MNIST dataset using ConvNet and ConvNet2(with residual connections).

# Installation & Environment Setup
```
conda create -n template python=3.12
conda activate template
pip install -r requirements.txt
```
# Train & Test
```
# Train
bash scripts/train_convnet.sh
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
