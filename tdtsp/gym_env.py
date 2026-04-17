import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .simulator import TDTSPSimulator
from .features import extract_global_features, extract_node_features

class TDTSPGymEnv(gym.Env):
    def __init__(self, simulator: TDTSPSimulator):
        super().__init__()
        self.sim = simulator
        n = self.sim.n
        
        self.action_space = spaces.Discrete(n)
        
        # Dict containing nodes features and global state along with the permissible actions
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(n, 4), dtype=np.float32),
            "global_features": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(n,), dtype=np.int8)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        return self._get_obs(), self._get_info()

    def step(self, action):
        _, reward, done, info = self.sim.step(int(action))
        
        if "error" in info:
            # Punish invalid moves explicitly in RL abstraction
            reward = -1e6
            done = True
        
        return self._get_obs(), reward, done, False, self._get_info()

    def _get_obs(self):
        return {
            "node_features": extract_node_features(self.sim),
            "global_features": extract_global_features(self.sim),
            "action_mask": self.sim.get_action_mask()
        }

    def _get_info(self):
        return {
            "tour": self.sim.tour.copy(),
            "cost": self.sim.cost,
            "time": self.sim.time,
            "visited_all": self.sim.visited.all()
        }
