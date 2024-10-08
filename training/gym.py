import training.simrunner as sr

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np

from training.simrunner import CombiSensor

MAX_WHEEL_SPEED = 7.0
MAX_VIEW_DISTANCE = 500.0

class SEnv(gym.Env):
    pass

# Define action and observation space

action_space = Box(low=-MAX_WHEEL_SPEED, high=MAX_WHEEL_SPEED, shape=(2,1), dtype=np.float32)

observation_view_space = Discrete(n=3)
observation_border_space = Box(low=0.0, high=MAX_VIEW_DISTANCE, shape=(3,1), dtype=np.float32)
observation_space = Dict({
    "view": observation_view_space,
    "border": observation_border_space,
})

def sensor_to_observation_space(sensor: sr.CombiSensor) -> gym.Space[observation_space]:
    pass

def action_space_to_diff_drive(action_space: gym.Space[action_space]) -> sr.DiffDriveValues:
    pass