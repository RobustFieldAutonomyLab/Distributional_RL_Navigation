import numpy as np
import sys
sys.path.insert(0,"./thirdparty")
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import gym

if __name__ == "__main__":
    
    map = gym.make('marine_env:marine_env-v0')

    map.init_visualize()
    map.visualize_control([1,1])