import sys
sys.path.insert(0,"./thirdparty")
from thirdparty import IQNAgent
import gym
import os
import argparse

parser = argparse.ArgumentParser(description="Train IQN model")
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

def run_trial(device):

    save_dir = "experiment_12_16"
    os.makedirs(save_dir)
    
    train_env = gym.make('marinenav_env:marinenav_env-v0')
    eval_env = gym.make('marinenav_env:marinenav_env-v0',seed=1)

    model = IQNAgent(train_env.get_state_space_dimension(),
                     train_env.get_action_space_dimension(),
                     device=device,
                     seed=2)

    model.learn(total_timesteps=4000000,
                train_env=train_env,
                eval_env=eval_env,
                eval_freq=10000,
                eval_log_path=save_dir)

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    args = parser.parse_args()

    if args.num_procs == 1:
        run_trial(args.device)
 