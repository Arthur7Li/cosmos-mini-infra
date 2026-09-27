"""
Distributed training utilities (DDP and FSDP) with a single-process fallback.
"""

import os
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

# Import the Block class to use for FSDP auto-wrap policy
from worldmodel.model import Block


def setup_distributed() -> tuple[bool, int, int, int]:
    """
    Initializes the process group if launched via torchrun.
    Returns (is_distributed, rank, local_rank, world_size).
    """
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if dist.is_available() and not dist.is_initialized():
            # Use NCCL for GPU, Gloo for CPU/MPS (if testing on Mac)
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def cleanup_distributed() -> None:
    """Destroys the process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: nn.Module, strategy: str, local_rank: int) -> nn.Module:
    """
    Wraps the model in DDP, FSDP, or returns it unmodified based on the strategy.

    Args:
        model: The PyTorch module to wrap.
        strategy: "none", "ddp", or "fsdp".
        local_rank: The local device ID for DDP wrapping.
    """
    if strategy == "none" or not dist.is_initialized():
        return model

    if strategy == "ddp":
        # If CUDA is available, we assign the device_id. For CPU (Gloo), device_ids must be None.
        if torch.cuda.is_available():
            return DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            return DDP(model)

    elif strategy == "fsdp":
        # Create an auto-wrap policy for our Transformer blocks
        # This keeps the blocks contiguous in memory instead of sharding them randomly
        my_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block},
        )
        return FSDP(model, auto_wrap_policy=my_auto_wrap_policy)

    else:
        raise ValueError(f"Unknown distributed strategy: {strategy}")
