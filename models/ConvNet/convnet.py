import torch
from torch import nn

from typing import List
from einops import rearrange

from .block import Block

class ConvNet(nn.Module):
    def __init__(self,
                 in_channels:int = 3, # Channels
                 layers:List = [8, 16, 32],
                 class_num:int = 10):
        super().__init__()
        
        dims = [in_channels, *layers]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.block_li = nn.ModuleList()
        for (in_dim, out_dim) in in_out:
            self.block_li.append(Block(in_dim=in_dim,
                                       out_dim=out_dim))
        
        self.li = nn.Linear(dims[-1], class_num)
        
    def forward(self, x):
        for block in self.block_li:
            x = block(x)
        
        x = rearrange(x, 'b c h w -> b c (h w)') # Flatten
        
        x = torch.mean(x, dim=2) # Global Average Pooling
        
        x = self.li(x)
        
        return x
    
    @classmethod
    def from_config(cls, cfg):
        return cls(in_channels=cfg['in_channels'],
                   layers=cfg['layers'],
                   class_num=cfg['class_num'])