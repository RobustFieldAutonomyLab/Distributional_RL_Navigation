# from tabnanny import verbose
import numpy as np
import sys
sys.path.insert(0,"./thirdparty")
from thirdparty import PPO
from thirdparty import A2C
from thirdparty import DQN
from thirdparty import QRDQN
import gym
import os
import argparse
import itertools
from multiprocessing import Pool
import json
from datetime import datetime
import marinenav_env.envs.marinenav_env as marinenav_env
import scipy.spatial

parser = argparse.ArgumentParser(description="Train sb3 model")

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
    default="auto",
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
    
    train_env = gym.make('marinenav_env:marinenav_env-v0',seed=params["seed"])
    

    eval_config = {}
    evaluate_env = gym.make('marinenav_env:marinenav_env-v0',seed=348)
    print("Creating 30 evaluation environments\n")
    eval_config = create_eval_configs(evaluate_env) 

    model = DQN(policy='ObsEncoderPolicy',
               env=train_env,
               learning_starts=10000,
               train_freq=1,
               verbose=1,
               seed=1,
               gamma=train_env.discount,
               device=device)
    
    # policy_args = {"n_quantiles":8}
    # model = QRDQN(policy='MlpPolicy',
    #               env=train_env,
    #               learning_starts=10000,
    #               policy_kwargs=policy_args,
    #               train_freq=1,
    #               exploration_fraction=0.1,
    #               exploration_initial_eps=1.0,
    #               exploration_final_eps=0.05,
    #               verbose=1,
    #               seed=1,
    #               gamma=train_env.discount,
    #               device=device)

    model.learn(total_timesteps=params["total_timesteps"],
                eval_env=evaluate_env,
                eval_config=eval_config,
                eval_freq=params["eval_freq"],
                n_eval_episodes=1, 
                eval_log_path=exp_dir)
    
    train_env.close()
    evaluate_env.close()

def create_eval_configs(eval_env):
    eval_config = {}

    eval_env.obs_r_range = [1,1]
    eval_env.num_obs = 6
    eval_env.start = np.array([6.0,6.0])
    eval_env.goal = np.array([44.0,44.0])

    for i in range(30): 
        eval_env.cores.clear()
        eval_env.obstacles.clear()

        num_cores = eval_env.num_cores
        num_obs = eval_env.num_obs

        # generate vortex with random position, spinning direction and strength
        if num_cores > 0:
            iteration = 500
            while True:
                center = eval_env.rd.uniform(low = np.zeros(2), high = np.array([eval_env.width,eval_env.height]))
                direction = eval_env.rd.binomial(1,0.5)
                v_edge = eval_env.rd.uniform(low = eval_env.v_range[0], high = eval_env.v_range[1])
                Gamma = 2 * np.pi * eval_env.r * v_edge
                core = marinenav_env.Core(center[0],center[1],direction,Gamma)
                iteration -= 1
                if eval_env.check_core(core):
                    eval_env.cores.append(core)
                    num_cores -= 1
                if iteration == 0 or num_cores == 0:
                    break
        
        centers = None
        for core in eval_env.cores:
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
    
        # KDTree storing vortex core center positions
        if centers is not None:
            eval_env.core_centers = scipy.spatial.KDTree(centers)

        # generate obstacles with random position in [10,40]
        if num_obs > 0:
            iteration = 500
            while True:
                center = eval_env.rd.uniform(low = 10*np.ones(2), high = 40*np.ones(2))
                r = 1.0
                obs = marinenav_env.Obstacle(center[0],center[1],r)
                iteration -= 1
                if eval_env.check_obstacle(obs):
                    eval_env.obstacles.append(obs)
                    num_obs -= 1
                if iteration == 0 or num_obs == 0:
                    break

        centers = None
        for obs in eval_env.obstacles:
            if centers is None:
                centers = np.array([[obs.x,obs.y]])
            else:
                c = np.array([[obs.x,obs.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing obstacle center positions
        if centers is not None: 
            eval_env.obs_centers = scipy.spatial.KDTree(centers)

        # reset robot state
        eval_env.reset_robot()

        # save eval config
        eval_config[f"env_{i}"] = eval_env.episode_data()

    return eval_config

if __name__ == "__main__":
    args = parser.parse_args()
    params = json.load(args.config_file)
    params_dashboard(params)
    trial_param_list = trial_params(params)

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    if args.num_procs == 1:
        for param in trial_param_list:
            param["training_time"]=timestamp
            run_trial(args.device,param)
    else:
        with Pool(processes=args.num_procs) as pool:
            for param in trial_param_list:
                param["training_time"]=timestamp
                pool.apply_async(run_trial,(args.device,param,))
            
            pool.close()
            pool.join()

    
