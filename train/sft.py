"""
Supervised Fine-Tuning (SFT) pretraining loop for the World Model.
"""

import argparse
import os
from collections.abc import Iterator

import gymnasium as gym
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

import envs  # noqa: F401
from worldmodel import ContinuousTokenizer, WorldModel, WorldModelConfig


class RandomTrajectoryDataset(IterableDataset):
    """
    Generates random trajectories from the environment on-the-fly.
    Yields (inputs, targets) where targets are inputs shifted by 1.
    """

    def __init__(
        self,
        env_id: str,
        tokenizer: ContinuousTokenizer,
        max_seq_len: int,
        steps_per_epoch: int,
    ):
        super().__init__()
        self.env_id = env_id
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        env = gym.make(self.env_id)

        for _ in range(self.steps_per_epoch):
            obs, _ = env.reset()
            tokens: list[int] = []

            # Collect enough tokens to fill max_seq_len + 1
            while len(tokens) < self.max_seq_len + 1:
                action = env.action_space.sample()
                next_obs, _, terminated, truncated, _ = env.step(action)

                step_tokens = self.tokenizer.encode(obs, action)
                tokens.extend(step_tokens.tolist())

                obs = next_obs
                if terminated or truncated:
                    obs, _ = env.reset()

            seq = torch.tensor(tokens[: self.max_seq_len + 1], dtype=torch.long)
            x = seq[:-1]
            y = seq[1:]
            yield x, y


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

    env = gym.make("ToyPhysics-v0")
    tokenizer = ContinuousTokenizer(
        env.observation_space, env.action_space, num_bins=config["worldmodel"]["vocab_size"]
    )

    wm_config = WorldModelConfig(**config["worldmodel"])
    model = WorldModel(wm_config).to(device)

    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"])

    epochs = 1 if args.smoke else config["train"]["epochs"]
    steps_per_epoch = 2 if args.smoke else config["train"]["steps_per_epoch"]
    batch_size = 2 if args.smoke else config["train"]["batch_size"]

    dataset = RandomTrajectoryDataset(
        "ToyPhysics-v0", tokenizer, wm_config.max_seq_len, steps_per_epoch * batch_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    use_amp = config["train"].get("amp", False)
    device_type = "cuda" if "cuda" in device else "cpu"

    scaler = torch.amp.GradScaler(device_type, enabled=use_amp and device_type == "cuda")

    model.train()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                _, loss = model(x, y)

            if loss is None:
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 10 == 0 or args.smoke:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")

    if not args.smoke:
        os.makedirs("output", exist_ok=True)
        torch.save(model.state_dict(), "output/worldmodel.pt")
        print("Saved checkpoint to output/worldmodel.pt")


if __name__ == "__main__":
    main()
