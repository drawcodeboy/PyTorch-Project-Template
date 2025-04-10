from .ConvNet.convnet import ConvNet
from .ConvNet2.convnet2 import ConvNet2

def load_model(cfg):
    if cfg['name'] == 'ConvNet':
        return ConvNet.from_config(cfg)

    elif cfg['name'] == 'ConvNet2':
        return ConvNet2.from_config(cfg)
        
    else:
        raise Exception(f"Model: {cfg['name']} is not supported.")