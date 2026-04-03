from datasets import load_dataset
from models import load_model

from utils import train_one_epoch, validate, save_ckpt, init_distributed_mode

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse, time, os, sys, yaml
import wandb

def add_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', action='store_true') # Resume from checkpoint last.ckpt
    parser.add_argument('--use_wandb', action='store_true')
    
    # Distributed Training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--init_method', type=str, default='env://')

    return parser

def load_wandb(cfg):
    wandb.init(
        config=cfg,
        project='Torch Template (MNIST Classification)',
        group=f"train_{cfg['model']['name']}_{cfg['data']['train']['dataset']}"
    )

    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

def set_seed(seed, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(cfg, args):
    WORLD_SIZE, LOCAL_RANK, RANK = None, None, None
    if args.distributed:
        WORLD_SIZE, LOCAL_RANK, RANK = init_distributed_mode(args, cfg)

    USE_WANDB = ((args.distributed==True and RANK == 0) and args.use_wandb) or (args.distributed==False and args.use_wandb)

    start_epoch = 1
    if args.resume == True:
        print("Resume Training")
        ckpt_path = os.path.join(cfg['ckpt_path'], "last.ckpt")
        ckpt = torch.load(ckpt_path, weights_only=False)
        cfg = ckpt['cfg']
        best_metric = ckpt['best_metric']
        start_epoch = ckpt['epoch'] + 1
        print(f"Load checkpoint from {ckpt_path}")

    # WandB Setting
    if USE_WANDB:
        # Only master process (RANK 0) will log to WandB in distributed training
        # Or, if not distributed, log to WandB as usual
        load_wandb(cfg)
    
    # Device Setting
    device = None
    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # Seed Setting
    set_seed(cfg['seed'], RANK if args.distributed else 0)

    # Hyperparameter Settings
    hp_cfg = cfg['hyperparameters']

    # Load Dataset
    data_cfg = cfg['data']

    train_ds = load_dataset(data_cfg['train'])
    val_ds = load_dataset(data_cfg['val'])

    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=True)
        val_sampler = torch.utils.data.DistributedSampler(
            val_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False, drop_last=False)
    else:
        train_sampler, val_sampler = None, None

    train_dl = torch.utils.data.DataLoader(train_ds,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           batch_size=hp_cfg['batch_size'],
                                           drop_last=True)

    val_dl = torch.utils.data.DataLoader(val_ds,
                                         sampler=val_sampler,
                                         batch_size=hp_cfg['batch_size'],
                                         drop_last=False)

    print(f"Load Train dataset {data_cfg['train']['dataset']}")
    print(f"Load Validation dataset {data_cfg['val']['dataset']}")
    print(f"Effective Batch Size: {hp_cfg['batch_size'] * (WORLD_SIZE if args.distributed else 1)}  (per GPU Batch Size: {hp_cfg['batch_size']})")

    # Load Model
    model_cfg = cfg['model']
    print(model_cfg['name'])
    model = load_model(model_cfg).to(device)

    if args.resume == True:
        model.load_state_dict(ckpt['model'])
        print(f"Load Model {model_cfg['name']} from checkpoint")

    if args.distributed and device == torch.device("cuda"):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])
    elif args.distributed and device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Loss Function
    if hp_cfg['loss_fn'] == 'cross-entropy':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Check loss function in configuration file")
    
    # Optimizer
    optimizer = None
    if hp_cfg['optim'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=hp_cfg['lr'])
    elif hp_cfg['optim'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=hp_cfg['lr'])
    
    # Load Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=7,
                                                     min_lr=1e-6)

    if args.resume == True:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        print(f"Load Optimizer and Scheduler from checkpoint")
    
    # Training loss
    total_start_time = int(time.time())

    ckpt_path = str(cfg['ckpt_path'])
    
    best_metric = 1e4
    
    for current_epoch in range(start_epoch, hp_cfg['epochs']+1):
        print("=======================================================")
        print(f"Epoch: [{current_epoch:03d}/{hp_cfg['epochs']:03d}]\n")

        if args.distributed:
            train_sampler.set_epoch(current_epoch)
        
        # Training One Epoch
        start_time = int(time.time())
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, scheduler, device)
        elapsed_time = int(time.time() - start_time)
        print(f"Train Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s")

        # Validation
        val_loss = validate(model, val_dl, loss_fn, device)

        if (val_loss < best_metric) and (RANK == 0 if args.distributed else True):
            best_metric = val_loss
            save_ckpt(ckpt_name="best",
                      model=model.module if args.distributed else model,
                      current_epoch=current_epoch,
                      best_metric=best_metric,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      cfg=cfg,
                      ckpt_path=ckpt_path)

        if USE_WANDB:
            # Only master process (RANK 0) will log to WandB in distributed training
            # Or, if not distributed, log to WandB as usual
            wandb.log({"epoch": current_epoch, 
                       "train_loss": train_loss, 
                       "val_loss": val_loss})

        if RANK == 0 if args.distributed else True:
            save_ckpt(ckpt_name="last",
                    model=model.module if args.distributed else model,
                    current_epoch=current_epoch,
                    best_metric=best_metric,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    ckpt_path=ckpt_path)

    total_elapsed_time = int(time.time()) - total_start_time
    print(f"<Total Train Time: {total_elapsed_time//60:02d}m {total_elapsed_time%60:02d}s>")

    if USE_WANDB:
        wandb.finish()

    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args, _ = parser.parse_known_args()

    with open(f'configs/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg, args)