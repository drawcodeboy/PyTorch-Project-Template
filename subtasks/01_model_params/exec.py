import os, sys
sys.path.append(os.getcwd())

from models import load_model

def main():
    model_cfg = {
        "name": "ConvNet",
        "in_channels": 1,
        "layers": [8, 16, 32],
        "class_num": 10
    }
    model = load_model(model_cfg)

    p_sum = 0
    for p in model.parameters():
        p_sum += p.numel()

    print(f"Model: {model_cfg['name']}, Number of parameters: {p_sum}")

if __name__ == '__main__':
    main()