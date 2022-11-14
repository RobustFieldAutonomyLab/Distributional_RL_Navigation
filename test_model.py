from tabnanny import verbose
import numpy as np
import sys
sys.path.insert(0,"./thirdparty")
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import gym
import os

def run_trial():
    
    save_dir = "experiment"
    os.makedirs(save_dir)
    
    train_env = gym.make('marinenav_env:marinenav_env-v0')
    evaluate_env = gym.make('marinenav_env:marinenav_env-v0',seed=1) 

    model = DQN(policy='MlpPolicy',
                env=train_env,
                train_freq=1,
                verbose=1,
                seed=0,
                gamma=train_env.discount)
            
    model.learn(total_timesteps=4000000,
                eval_env=evaluate_env,
                eval_freq=10000,
                n_eval_episodes=1, 
                eval_log_path=save_dir)
    
    train_env.close()
    evaluate_env.close()

if __name__ == "__main__":
    
    run_trial()

    