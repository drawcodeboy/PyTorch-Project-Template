import torch
import torch.nn.functional as F

import numpy as np

from .metrics import get_metrics

PER_STEPS = 10

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = []
    
    for batch_idx, (x, target) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        x = x.to(device)
        target = target.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, target)
        
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % PER_STEPS == 0:
            print(f"Training steps [{batch_idx}/{len(dataloader)}]: Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    scheduler.step(sum(total_loss)/len(total_loss))
    print(f"Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return sum(total_loss)/len(total_loss)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    
    total_outputs = []
    total_targets = []
    
    for batch_idx, (x, target) in enumerate(dataloader, start=1):
        x = x.to(device)
        target = target.to(device)
        
        logits = model(x)
        out = F.softmax(logits, dim=1)
        out = torch.argmax(out, dim=1)
        
        total_outputs.extend(out.tolist())
        total_targets.extend(target.tolist())
        
        if batch_idx % PER_STEPS == 0:
            print(f"Evaluate steps [{batch_idx}/{len(dataloader)}]")
    print()
    
    result = get_metrics(np.array(total_outputs), np.array(total_targets))
    
    return result