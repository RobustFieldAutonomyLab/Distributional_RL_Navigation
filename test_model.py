from tabnanny import verbose
import numpy as np
import sys
sys.path.insert(0,"./thirdparty")
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import gym
import os
import argparse

parser = argparse.ArgumentParser(description="Run baseline experiments")
parser.add_argument(
    "-P",
    "--num-procs",
    dest="num_procs",
    type=int,
    default=1,
    help="number of subprocess workers to use for trial parallelization",
)
parser.add_argument(
    "-D",
    "--device",
    dest="device",
    type=str,
    default="auto",
    help="device to run all subprocesses, could only specify 1 device in each run"
)

def run_trial():
    
    save_dir = "experiment_dist_reward_energy_penalty"
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
    args = parser.parse_args()

    if args.num_procs == 1:
        run_trial()

    