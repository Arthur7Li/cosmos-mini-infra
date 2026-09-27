"""
Discretizes continuous state and action vectors into integer tokens.
"""

import gymnasium as gym
import numpy as np
import torch


class ContinuousTokenizer:
    """
    A simple tokenizer that uses linear binning to discretize continuous
    ranges into integer tokens for transformer consumption.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, num_bins: int = 256):
        self.num_bins = num_bins

        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("Observation space must be a Box.")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Action space must be a Box.")

        self.s_low = observation_space.low
        self.s_high = observation_space.high
        self.a_low = action_space.low
        self.a_high = action_space.high

        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]

    def _discretize(self, value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Map a continuous value in [low, high] to an integer in [0, num_bins-1]."""
        value = np.clip(value, low, high)
        span = high - low
        span[span == 0] = 1.0  # avoid division by zero
        normalized = (value - low) / span
        bins = np.round(normalized * (self.num_bins - 1)).astype(np.int32)
        return bins

    def _undiscretize(self, bins: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Map an integer in [0, num_bins-1] back to the center of its continuous bin."""
        normalized = bins.astype(np.float32) / (self.num_bins - 1)
        span = high - low
        return low + normalized * span

    def encode(self, state: np.ndarray, action: np.ndarray) -> torch.Tensor:
        """
        Encode state and action arrays into a single flat tensor of token IDs.
        """
        s_bins = self._discretize(state, self.s_low, self.s_high)
        a_bins = self._discretize(action, self.a_low, self.a_high)

        seq = np.concatenate([s_bins, a_bins])
        return torch.tensor(seq, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode a tensor of token IDs back into approximate state and action arrays.
        """
        seq = tokens.cpu().numpy()

        s_bins = seq[: self.state_dim]
        a_bins = seq[self.state_dim : self.state_dim + self.action_dim]

        state = self._undiscretize(s_bins, self.s_low, self.s_high)
        action = self._undiscretize(a_bins, self.a_low, self.a_high)

        return state, action
