"""Tests for the ContinuousTokenizer."""

import gymnasium as gym
import numpy as np
import torch

from worldmodel.tokenizer import ContinuousTokenizer


def test_continuous_tokenizer() -> None:
    """Test that encode and decode map continuous values back to themselves within tolerance."""
    obs_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    tokenizer = ContinuousTokenizer(obs_space, act_space, num_bins=256)

    # Pick some arbitrary state and action inside the bounds
    original_state = np.array([2.5, -0.5], dtype=np.float32)
    original_action = np.array([0.25], dtype=np.float32)

    # Encode to discrete tokens
    tokens = tokenizer.encode(original_state, original_action)

    assert tokens.shape == (3,)
    assert tokens.dtype == torch.long
    assert (tokens >= 0).all() and (tokens < 256).all()

    # Decode back to continuous values
    dec_state, dec_action = tokenizer.decode(tokens)

    # The max error is 0.5 * span / (num_bins - 1)
    state_span = obs_space.high - obs_space.low
    action_span = act_space.high - act_space.low

    state_tolerance = (state_span / 255).max()
    action_tolerance = (action_span / 255).max()

    np.testing.assert_allclose(original_state, dec_state, atol=state_tolerance)
    np.testing.assert_allclose(original_action, dec_action, atol=action_tolerance)
