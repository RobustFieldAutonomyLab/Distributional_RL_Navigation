import sys
sys.path.insert(0,"./thirdparty")
from thirdparty import IQNAgent
import gym
import os
import argparse
import itertools
from multiprocessing import Pool
import json
from datetime import datetime
import numpy as np
import marinenav_env.envs.marinenav_env as marinenav_env
import scipy.spatial

parser = argparse.ArgumentParser(description="Train IQN model")

parser.add_argument(
    "-C",
    "--config-file",
    dest="config_file",
    type=open,
    required=True,
    help="configuration file for training parameters",
)
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
    default="cpu",
    help="device to run all subprocesses, could only specify 1 device in each run"
)

def product(*args, repeat=1):
    # This function is a modified version of 
    # https://docs.python.org/3/library/itertools.html#itertools.product
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def trial_params(params):
    if isinstance(params,(str,int,float)):
        return [params]
    elif isinstance(params,list):
        return params
    elif isinstance(params, dict):
        keys, vals = zip(*params.items())
        mix_vals = []
        for val in vals:
            val = trial_params(val)
            mix_vals.append(val)
        return [dict(zip(keys, mix_val)) for mix_val in itertools.product(*mix_vals)]
    else:
        raise TypeError("Parameter type is incorrect.")

def params_dashboard(params):
    print("\n====== Training Setup ======\n")
    print("seed: ",params["seed"])
    print("total_timesteps: ",params["total_timesteps"])
    print("eval_freq: ",params["eval_freq"])
    print("\n")

def run_trial(device,params):

    exp_dir = os.path.join(params["save_dir"],
                           "training_"+params["training_time"],
                           "seed_"+str(params["seed"]))
    os.makedirs(exp_dir)

    param_file = os.path.join(exp_dir,"trial_config.json")
    with open(param_file, 'w+') as outfile:
        json.dump(params, outfile)

    # schedule of curriculum training
    training_schedule = dict(timesteps=[0,1000000,2000000],
                             num_cores=[4,6,8],
                             num_obstacles=[6,8,10],
                             min_start_goal_dis=[30.0,35.0,40.0],
                             )
    
    schedule_file = os.path.join(exp_dir,"training_schedule.json")
    with open(schedule_file, 'w+') as outfile:
        json.dump(training_schedule, outfile)

    train_env = gym.make('marinenav_env:marinenav_env-v0',seed=params["seed"],schedule=training_schedule)
    
    # evaluation environment configs
    eval_config = {}
    eval_env = gym.make('marinenav_env:marinenav_env-v0',seed=348)
    print("Creating 30 evaluation environments\n")
    eval_config = create_eval_configs(eval_env)

    eval_config_file = os.path.join(exp_dir,"eval_config.json")
    with open(eval_config_file, "w+") as f:
        json.dump(eval_config, f)

    model = IQNAgent(train_env.get_state_space_dimension(),
                     train_env.get_action_space_dimension(),
                     device=device,
                     seed=params["seed"]+100)

    model.learn(total_timesteps=params["total_timesteps"],
                train_env=train_env,
                eval_env=eval_env,
                eval_config=eval_config,
                eval_freq=params["eval_freq"],
                eval_log_path=exp_dir)

    train_env.close()
    eval_env.close()

def create_eval_configs(eval_env):
    eval_config = {}

    # varying the number of cores and obstacles to adjust the level of difficulty
    num_episodes = [10,10,10]
    num_cs = [4,6,8]
    num_os = [6,8,10]

    eval_env.obs_r_range = [1,3]
    eval_env.reset_start_and_goal = False
    eval_env.start = np.array([5.0,5.0])
    eval_env.goal = np.array([45.0,45.0])

    count = 0
    for i,num_episode in enumerate(num_episodes):
        for _ in range(num_episode): 
            eval_env.num_cores = num_cs[i]
            eval_env.num_obs = num_os[i]

            eval_env.reset()

            # save eval config
            eval_config[f"env_{count}"] = eval_env.episode_data()
            count += 1

    return eval_config

if __name__ == "__main__":
    args = parser.parse_args()
    params = json.load(args.config_file)
    params_dashboard(params)
    trial_param_list = trial_params(params)

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    # create and save 30 randomly generated evaluation environments config
    # eval_config = {}
    # env = gym.make('marinenav_env:marinenav_env-v0',seed=348)
    # print("Creating 30 evaluation environments\n")
    # # for i in range(30):
    # #     env.reset()
    # #     eval_config[f"env_{i}"] = env.episode_data()
    # eval_config = create_eval_configs(env)

    if args.num_procs == 1:
        for param in trial_param_list:
            param["training_time"]=timestamp
            run_trial(args.device,param)
    else:
        with Pool(processes=args.num_procs) as pool:
            for param in trial_param_list:
                param["training_time"]=timestamp
                pool.apply_async(run_trial,(args.device,param))
            
            pool.close()
            pool.join()

 