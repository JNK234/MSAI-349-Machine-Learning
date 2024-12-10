import gymnasium as gym
import numpy as np

class SlotMachines(gym.Env):
    def __init__(self, n_machines=10):
        self.n_machines = n_machines
        self.payouts = np.random.uniform(0, 1, n_machines)
        self.action_space = gym.spaces.Discrete(n_machines)
        self.observation_space = gym.spaces.Discrete(1)  # Only one state
    
    def step(self, action):
        reward = np.random.binomial(1, self.payouts[action])
        return 0, reward, False, False, {}  # state, reward, terminated, truncated, info
    
    def reset(self):
        return 0, {}  # state, info
        
    def render(self):
        pass