"""
A simple 1D physics environment for trajectory generation.
"""
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ToyPhysicsEnv(gym.Env):
    """
    A simple 1D continuous control environment.

    The goal is to move a point mass to the origin (x=0) and keep it there.
    State is [position, velocity].
    Action is a 1D continuous force applied to the mass.
    """

    from typing import ClassVar
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # State: [position, velocity]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0], dtype=np.float32), 
            high=np.array([10.0, 10.0], dtype=np.float32), 
            dtype=np.float32
        )
        # Action: [force]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.dt = 0.05
        self.max_steps = 100
        self.current_step = 0

        self.state: np.ndarray = np.zeros(2, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize with random position and velocity
        position = self.np_random.uniform(low=-5.0, high=5.0)
        velocity = self.np_random.uniform(low=-1.0, high=1.0)
        self.state = np.array([position, velocity], dtype=np.float32)

        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1

        position, velocity = self.state
        force = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])

        # Physics update
        new_velocity = velocity + force * self.dt
        new_position = position + new_velocity * self.dt

        self.state = np.array([new_position, new_velocity], dtype=np.float32)

        # Reward: penalize distance from origin and large control forces
        reward = float(-(new_position**2) - 0.1 * (force**2))

        terminated = False
        truncated = self.current_step >= self.max_steps

        return self.state.copy(), reward, terminated, truncated, {}
