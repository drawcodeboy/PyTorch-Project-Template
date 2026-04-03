import torch
import os
import builtins
import datetime

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (torch.distributed.get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_distributed_mode(args, cfg):
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    
    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        # Even if you set cuda.set_device(LOCAL_RANK), you need to explicitly specify the device_id
        # in init_process_group to ensure that each process uses the correct GPU. 
        # Otherwise, all processes might try to use the default GPU (0) by default.
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method,
                                             world_size=WORLD_SIZE, rank=RANK, device_id=LOCAL_RANK)
    else:
        torch.distributed.init_process_group(backend="gloo", init_method=args.init_method,
                                             world_size=WORLD_SIZE, rank=RANK)
    
    torch.distributed.barrier()
    setup_for_distributed(RANK == 0)

    return WORLD_SIZE, LOCAL_RANK, RANK