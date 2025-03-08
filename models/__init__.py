from .ConvNet.convnet import ConvNet
from .ConvNet2.convnet2 import ConvNet2

def load_model(**cfg):
    if cfg['name'] == 'ConvNet':
        return ConvNet(in_channels=cfg['in_channels'],
                       layers=cfg['layers'],
                       class_num=cfg['class_num'])

    elif cfg['name'] == 'ConvNet2':
        return ConvNet2(in_channels=cfg['in_channels'],
                        layers=cfg['layers'],
                        class_num=cfg['class_num'])
        
    else:
        raise Exception(f"Model: {cfg['name']} is not supported.")