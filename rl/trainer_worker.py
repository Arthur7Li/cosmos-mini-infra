import argparse
import os

import torch
import torch.multiprocessing as mp
import yaml
from torch.optim import AdamW

from rl.rollout_worker import rollout_loop
from rl.weight_sync import AsyncWeightManager, SyncWeightManager
from worldmodel import WorldModel, WorldModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke test and exit")
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

    epochs = 1 if args.smoke else config["train"]["epochs"]
    steps_per_epoch = 2 if args.smoke else config["train"]["steps_per_epoch"]
    batch_size = 2 if args.smoke else config["train"]["batch_size"]

    use_amp = config["train"].get("amp", False)
    device_type = "cuda" if "cuda" in device else "cpu"

    scaler = torch.amp.GradScaler(device_type, enabled=use_amp and device_type == "cuda")

    # Set up queues
    ctx = mp.get_context("spawn")
    traj_queue = ctx.Queue()
    weight_queue = ctx.Queue()
    
    sync_mode = config.get("rl", {}).get("sync_mode", "sync")
    if sync_mode == "sync":
        weight_manager = SyncWeightManager(weight_queue)
    else:
        weight_manager = AsyncWeightManager(weight_queue)

    # Spawn rollout worker
    rollout_proc = ctx.Process(
        target=rollout_loop,
        args=(
            "ToyPhysics-v0",
            config["worldmodel"],
            traj_queue,
            weight_queue,
            sync_mode,
            100000000, # run indefinitely until terminated
            batch_size,
        )
    )
    rollout_proc.start()

    model.train()
    
    # Push initial weights to break the sync deadlock
    weight_manager.push_weights(model.state_dict())
    
    try:
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                # Read a batch of trajectories
                trajectories = []
                while len(trajectories) < batch_size:
                    # Blocking get
                    traj = traj_queue.get()
                    if traj is None:
                        break
                    trajectories.append(traj)

                if len(trajectories) < batch_size:
                    print("Not enough trajectories. Exiting early.")
                    break

                x = torch.stack([t[0] for t in trajectories]).to(device)
                y = torch.stack([t[1] for t in trajectories]).to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    _, loss = model(x, y)

                if loss is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if step % 10 == 0 or args.smoke:
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

                # Push weights to rollout worker
                weight_manager.push_weights(model.state_dict())

    finally:
        # Graceful termination
        rollout_proc.terminate()
        rollout_proc.join()

    if not args.smoke:
        os.makedirs("output", exist_ok=True)
        torch.save(model.state_dict(), "output/rl_worldmodel.pt")
        print("Saved checkpoint to output/rl_worldmodel.pt")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
