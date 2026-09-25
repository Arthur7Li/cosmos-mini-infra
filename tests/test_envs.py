"""Tests for the toy_physics environment."""
import numpy as np

from envs.toy_physics import ToyPhysicsEnv


def test_toy_physics_env() -> None:
    """Test the ToyPhysicsEnv complies with the Gym API and is seedable."""
    env = ToyPhysicsEnv()

    # Test reset and spaces
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    # Test step
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)


def test_toy_physics_reproducibility() -> None:
    """Test that seeding the environment produces reproducible states."""
    env1 = ToyPhysicsEnv()
    obs1, _ = env1.reset(seed=42)

    env2 = ToyPhysicsEnv()
    obs2, _ = env2.reset(seed=42)

    np.testing.assert_array_equal(obs1, obs2)

    action = np.array([0.5], dtype=np.float32)
    next_obs1, *_ = env1.step(action)
    next_obs2, *_ = env2.step(action)

    np.testing.assert_array_equal(next_obs1, next_obs2)
