"""
Profiles the RL synchronization modes using torch.profiler and records total throughput.
"""
import argparse
import os
import time

import torch
import torch.multiprocessing as mp
import yaml
from torch.optim import AdamW
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from rl.rollout_worker import rollout_loop
from rl.weight_sync import AsyncWeightManager, SyncWeightManager
from worldmodel import WorldModel, WorldModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config["train"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    wm_config = WorldModelConfig(**config["worldmodel"])
    model = WorldModel(wm_config).to(device)

    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"])

    # For profiling, we fix the number of steps to a small amount to keep trace sizes manageable
    steps_per_epoch = 10 
    batch_size = config["train"]["batch_size"]

    use_amp = config["train"].get("amp", False)
    device_type = "cuda" if "cuda" in device else "cpu"

    scaler = torch.amp.GradScaler(device_type, enabled=use_amp and device_type == "cuda")

    ctx = mp.get_context("spawn")
    traj_queue = ctx.Queue()
    weight_queue = ctx.Queue()

    sync_mode = config.get("rl", {}).get("sync_mode", "sync")
    if sync_mode == "sync":
        weight_manager = SyncWeightManager(weight_queue)
    else:
        weight_manager = AsyncWeightManager(weight_queue)

    rollout_proc = ctx.Process(
        target=rollout_loop,
        args=(
            "ToyPhysics-v0",
            config["worldmodel"],
            traj_queue,
            weight_queue,
            sync_mode,
            100000000,
            batch_size,
        )
    )
    rollout_proc.start()
    model.train()
    weight_manager.push_weights(model.state_dict())

    os.makedirs("output/traces", exist_ok=True)
    trace_dir = f"output/traces/rl_{sync_mode}"

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"Starting Profiling for mode: {sync_mode}...")
    
    start_time = time.perf_counter()

    try:
        with profile(
            activities=activities,
            record_shapes=True,
            on_trace_ready=tensorboard_trace_handler(trace_dir),
        ) as prof:
            
            for step in range(steps_per_epoch):
                with torch.profiler.record_function("wait_for_trajectories"):
                    trajectories = []
                    while len(trajectories) < batch_size:
                        traj = traj_queue.get()
                        if traj is None:
                            break
                        trajectories.append(traj)

                with torch.profiler.record_function("forward_backward_step"):
                    x = torch.stack([t[0] for t in trajectories]).to(device)
                    y = torch.stack([t[1] for t in trajectories]).to(device)

                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                        _, loss = model(x, y)

                    if loss is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                with torch.profiler.record_function("push_weights"):
                    weight_manager.push_weights(model.state_dict())

                prof.step()
                
    finally:
        rollout_proc.terminate()
        rollout_proc.join()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_samples = steps_per_epoch * batch_size
    samples_per_sec = total_samples / total_time
    
    print(f"--- Profiling Complete: {sync_mode} ---")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"Trace saved to: {trace_dir}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
