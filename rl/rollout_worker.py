"""
Rollout worker for collecting trajectories from the environment.
"""

import gymnasium as gym
import torch

import envs  # noqa: F401

from rl.weight_sync import AsyncWeightManager, SyncWeightManager
from worldmodel.model import WorldModel, WorldModelConfig
from worldmodel.tokenizer import ContinuousTokenizer


def rollout_loop(
    env_id: str,
    model_config: dict,
    traj_queue,
    weight_queue,
    sync_mode: str,
    num_steps: int,
    batch_size: int = 16,
):
    """
    Main loop for collecting trajectories.
    """
    # Instantiate environment and model
    env = gym.make(env_id)
    wm_config = WorldModelConfig(**model_config)
    model = WorldModel(wm_config)

    # Initialize tokenizer
    tokenizer = ContinuousTokenizer(env.observation_space, env.action_space, num_bins=wm_config.vocab_size)

    # Instantiate weight manager
    if sync_mode == "sync":
        manager = SyncWeightManager(weight_queue)
    elif sync_mode == "async":
        manager = AsyncWeightManager(weight_queue)
    else:
        raise ValueError(f"Unknown sync_mode: {sync_mode}")

    obs, _ = env.reset()
    tokens = []
    trajectories_collected = 0

    # Get initial weights
    weights = manager.get_weights()
    if weights is not None:
        model.load_state_dict(weights)

    for _ in range(num_steps):
        # Take a random step (in real RL, we would sample from model)
        action = env.action_space.sample()

        # Encode state and action
        step_tokens = tokenizer.encode(obs, action)
        tokens.extend(step_tokens.tolist())

        # Step environment
        next_obs, _reward, terminated, truncated, _ = env.step(action)
        obs = next_obs
        
        # Handle reset
        if terminated or truncated:
            obs, _ = env.reset()

        if len(tokens) >= wm_config.max_seq_len + 1:
            seq = torch.tensor(tokens[:wm_config.max_seq_len + 1], dtype=torch.long)
            x = seq[:-1]
            y = seq[1:]
            traj_queue.put((x, y))
            tokens = [] # reset for next sequence
            trajectories_collected += 1
            
            # Sync weights after completing a batch of trajectories
            if trajectories_collected % batch_size == 0:
                weights = manager.get_weights()
                if weights is not None:
                    model.load_state_dict(weights)
