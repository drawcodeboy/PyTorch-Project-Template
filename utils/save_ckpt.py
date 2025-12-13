import torch
import numpy as np
import os

def save_model_ckpt(model, model_name, current_epoch, save_dir, logger):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['epochs'] = current_epoch
    
    save_name = f"{model_name}.epochs_{current_epoch:03d}.pth"
    
    try:
        torch.save(ckpt, os.path.join(save_dir, save_name))
        logger.info(f"Save Model @epoch: {current_epoch}")
    except:
        logger.info(f"Can\'t Save Model @epoch: {current_epoch}")
        
def save_loss_ckpt(model_name, train_loss, save_dir, logger):
    try:
        np.save(os.path.join(save_dir, f'train_loss_{model_name}.npy'), np.array(train_loss))
        logger.info('Save Train Loss')
    except:
        logger.info('Can\'t Save Train Loss') 