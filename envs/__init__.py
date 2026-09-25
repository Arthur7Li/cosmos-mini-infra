"""Environment registration and exports."""
import gymnasium as gym

from envs.toy_physics import ToyPhysicsEnv

gym.register(
    id="ToyPhysics-v0",
    entry_point="envs.toy_physics:ToyPhysicsEnv",
)

__all__ = ["ToyPhysicsEnv"]
