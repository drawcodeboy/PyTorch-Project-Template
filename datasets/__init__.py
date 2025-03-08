from .mnist_dataset import MNIST_Dataset

def load_dataset(**cfg):
    if cfg['dataset'] == 'MNIST':
        return MNIST_Dataset(root=cfg['root'],
                             download=cfg['download'],
                             mode=cfg['mode'])
    
    else:
        raise Exception(f"Dataset: {cfg['dataset']} is not supported.")