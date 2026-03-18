import torch
import numpy as np
import os

def save_ckpt(ckpt_name, 
              model, 
              current_epoch, 
              best_metric, 
              optimizer, 
              scheduler, 
              cfg, 
              ckpt_path):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch,
        'best_metric': best_metric,
        'cfg': cfg
    }

    try:
        torch.save(ckpt, os.path.join(ckpt_path, f"{ckpt_name}.ckpt"))
        print(f"Checkpoint {ckpt_name}.ckpt saved successfully.")
    except:
        print(f"Can\'t Save Checkpoint {ckpt_name}.ckpt")