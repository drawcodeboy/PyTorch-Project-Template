import torch
import os

def init_distributed_mode():
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    print(f"RANK: {RANK}, WORLD_SIZE: {WORLD_SIZE}, LOCAL_RANK: {LOCAL_RANK}")
    
    torch.distributed.barrier()
    exit()