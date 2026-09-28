"""
Benchmark script for comparing single vs multi-process throughput.
"""

import argparse
import contextlib
import json
import os
import time
from typing import Any

import torch
from torch.optim import AdamW

from train.distributed import cleanup_distributed, setup_distributed, wrap_model
from worldmodel.model import WorldModel, WorldModelConfig


def benchmark_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    use_no_sync: bool,
) -> float:
    """Runs a single forward/backward/step and returns the elapsed time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # Optional no_sync context to simulate lack of communication overlap (only for DDP)
    ctx = model.no_sync() if use_no_sync and hasattr(model, "no_sync") else contextlib.nullcontext()

    with ctx:
        optimizer.zero_grad()
        _, loss = model(x, y)
        if loss is not None:
            loss.backward()

    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter() - start_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", type=str, default="none", choices=["none", "ddp", "fsdp"]
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--no-overlap", action="store_true", help="Disable communication overlap (DDP only)"
    )
    args = parser.parse_args()

    is_dist, rank, local_rank, world_size = setup_distributed()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if is_dist and device == "cuda":
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)

    # Instantiate model
    config = WorldModelConfig(max_seq_len=args.seq_len)
    model = WorldModel(config).to(device)
    model = wrap_model(model, args.strategy, local_rank)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Dummy data
    x = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)

    model.train()

    # Warmup
    for _ in range(args.warmup):
        benchmark_step(model, optimizer, x, y, args.no_overlap)

    # Benchmark
    total_time = 0.0
    for _ in range(args.steps):
        step_time = benchmark_step(model, optimizer, x, y, args.no_overlap)
        total_time += step_time

    avg_step_time = total_time / args.steps
    samples_per_sec = args.batch_size * world_size / avg_step_time
    tokens_per_sec = samples_per_sec * args.seq_len

    metrics: dict[str, Any] = {
        "strategy": args.strategy,
        "world_size": world_size,
        "no_overlap": args.no_overlap,
        "avg_step_time_sec": avg_step_time,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
    }

    if rank == 0:
        print(f"Benchmark Results: {metrics}")
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", "metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    cleanup_distributed()


if __name__ == "__main__":
    main()
