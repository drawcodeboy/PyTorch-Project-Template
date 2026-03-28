import torch
import os

def init_distributed_mode():
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    print(f"RANK: {RANK}, WORLD_SIZE: {WORLD_SIZE}, LOCAL_RANK: {LOCAL_RANK}")
    
    # Backend & Init method should be set ... by args? config?
    torch.distributed.init_process_group(backend="gloo", init_method="env://")
    torch.distributed.barrier()
    exit()