import sys
sys.path.insert(0,"./thirdparty")
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from thirdparty import QRDQN
from thirdparty import IQNAgent
import APF
import BA
import os
import gym
import marinenav_env.envs.marinenav_env as marinenav_env
import numpy as np
import copy
import scipy.spatial
import env_visualizer
import json
from datetime import datetime
import time as t_module

def evaluation_IQN(first_observation, agent, test_env, adaptive:bool=False, cvar=1.0):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0
    
    quantiles_data = []
    taus_data = []

    cvars = []
    computation_times = []

    while not done and length < 1000:
        action = None
        if adaptive:
            start = t_module.time()
            (action, quantiles, taus), cvar = agent.act_adaptive_eval(observation)
            end = t_module.time()
            computation_times.append(end-start)
            cvars.append(cvar)
        else:
            start = t_module.time()
            action, quantiles, taus = agent.act_eval(observation,cvar=cvar)
            end = t_module.time()
            computation_times.append(end-start)
            cvars.append(cvar)

        quantiles_data.append(quantiles)
        taus_data.append(taus)

        # if len(quantiles_data) < len(cvars):
        #     quantiles_data.append(quantiles)
        #     taus_data.append(taus)
        # else:
        #     quantiles_data[i] = np.concatenate((quantiles_data[i],quantiles))
        #     taus_data[i] = np.concatenate((taus_data[i],taus))
        
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))
        
    # metric data
    success = True if info["state"] == "reach goal" else False
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length

    ep_data = test_env.episode_data()
    ep_data["robot"]["actions_cvars"] = copy.deepcopy(cvars)
    ep_data["robot"]["actions_quantiles"] = [x.tolist() for x in quantiles_data]
    ep_data["robot"]["actions_taus"] = [x.tolist() for x in taus_data]

    return ep_data, success, time, energy, out_of_area, computation_times

def evaluation_DQN(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0

    computation_times = []

    while not done and length < 1000:
        start = t_module.time()
        action, _ = agent.predict(observation,deterministic=True)
        end = t_module.time()
        computation_times.append(end-start)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))
        
    # metric data
    success = True if info["state"] == "reach goal" else False
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length

    return test_env.episode_data(), success, time, energy, out_of_area, computation_times

def evaluation_classical(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0

    computation_times = []
    
    while not done and length < 1000:
        start = t_module.time()
        action = agent.act(observation)
        end = t_module.time()
        computation_times.append(end-start)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))

    # metric data
    success = True if info["state"] == "reach goal" else False
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length

    return test_env.episode_data(), success, time, energy, out_of_area, computation_times

def exp_setup_1(envs):
    # keep default env settings
    pass

def exp_setup_2(envs):
    # fix the obstacle r = 0.5, and increases num to 10
    for env in envs:
        env.obs_r_range = [1,1]
        env.num_obs = 10

def exp_setup_3(envs,n_obs,n_cores):
    # 1. fix the obstacle r = 0.5, and increases num to 8
    # 2. reduce the area of obstacle generation to make them more dense
    # 3. fix the start and goal position so that obstacles lie in the line between them
    observations = []
    for env in envs:
        env.obs_r_range = [1,1]
        env.num_obs = n_obs
        env.num_cores = n_cores
        env.start = np.array([6.0,6.0])
        env.goal = np.array([44.0,44.0])

        # reset the environment
        env.cores.clear()
        env.obstacles.clear()

        num_cores = env.num_cores
        num_obs = env.num_obs

        # generate vortex with random position, spinning direction and strength
        if num_cores > 0:
            iteration = 500
            while True:
                center = env.rd.uniform(low = np.zeros(2), high = np.array([env.width,env.height]))
                direction = env.rd.binomial(1,0.5)
                v_edge = env.rd.uniform(low = env.v_range[0], high = env.v_range[1])
                Gamma = 2 * np.pi * env.r * v_edge
                core = marinenav_env.Core(center[0],center[1],direction,Gamma)
                iteration -= 1
                if env.check_core(core):
                    env.cores.append(core)
                    num_cores -= 1
                if iteration == 0 or num_cores == 0:
                    break
        
        centers = None
        for core in env.cores:
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing vortex core center positions
        if centers is not None:
            env.core_centers = scipy.spatial.KDTree(centers)

        # generate obstacles with random position and size
        if num_obs > 0:
            iteration = 500
            while True:
                center = env.rd.uniform(low = 10*np.ones(2), high = 40*np.ones(2))
                r = env.rd.uniform(low = env.obs_r_range[0], high = env.obs_r_range[1])
                obs = marinenav_env.Obstacle(center[0],center[1],r)
                iteration -= 1
                if env.check_obstacle(obs):
                    env.obstacles.append(obs)
                    num_obs -= 1
                if iteration == 0 or num_obs == 0:
                    break

        centers = None
        for obs in env.obstacles:
            if centers is None:
                centers = np.array([[obs.x,obs.y]])
            else:
                c = np.array([[obs.x,obs.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing obstacle center positions
        if centers is not None: 
            env.obs_centers = scipy.spatial.KDTree(centers)

        # reset robot state
        env.reset_robot()

        observations.append(env.get_observation())

    return observations

def demonstration(envs):
    # Demonstrate that RL agents are clearly better in adverse flow field
    observations = []

    for test_env in envs:
        test_env.cores.clear()
        test_env.obstacles.clear()
        
        # set start and goal
        test_env.start = np.array([15.0,10.0])
        test_env.goal = np.array([45.0,35.0])

        # set vortex cores data
        core_0 = marinenav_env.Core(14.0,1.0,0,np.pi*10.0)
        core_1 = marinenav_env.Core(10.0,18.0,0,np.pi*7.0)
        core_2 = marinenav_env.Core(15.0,26.0,1,np.pi*8.0)
        core_3 = marinenav_env.Core(25.0,23.0,1,np.pi*10.0)
        core_4 = marinenav_env.Core(13.0,41.0,0,np.pi*8.0)
        core_5 = marinenav_env.Core(40.0,22.0,0,np.pi*8.0)
        core_6 = marinenav_env.Core(36.0,30.0,0,np.pi*7.0)
        core_7 = marinenav_env.Core(37.0,37.0,1,np.pi*6.0)

        test_env.cores = [core_0,core_1,core_2,core_3, \
                        core_4,core_5,core_6,core_7]

        centers = None
        for core in test_env.cores:
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        if centers is not None:
            test_env.core_centers = scipy.spatial.KDTree(centers)

        # set obstacles
        obs_1 = marinenav_env.Obstacle(20.0,36.0,1.5)
        obs_2 = marinenav_env.Obstacle(35.0,19.0,1.5)
        obs_3 = marinenav_env.Obstacle(8.0,25.0,1.5)
        obs_4 = marinenav_env.Obstacle(30,33.0,1.5)

        test_env.obstacles = [obs_1,obs_2,obs_3,obs_4]

        centers = None
        for obs in test_env.obstacles:
            if centers is None:
                centers = np.array([[obs.x,obs.y]])
            else:
                c = np.array([[obs.x,obs.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing obstacle center positions
        if centers is not None: 
            test_env.obs_centers = scipy.spatial.KDTree(centers)

        # reset robot
        test_env.robot.init_theta = 3 * np.pi / 4
        test_env.robot.init_speed = 1.0
        current_v = test_env.get_velocity(test_env.start[0],test_env.start[1])
        test_env.robot.reset_state(test_env.start[0],test_env.start[1], current_velocity=current_v)

        observations.append(test_env.get_observation())

    return observations

def exp_setup_5(envs,n_obs,n_cores):
    # fix start and goal location, random vortexes and obstacles
    observations = []

    for test_env in envs:
        test_env.reset_start_and_goal = False
        test_env.random_reset_state = False
        test_env.set_boundary = True
        test_env.obs_r_range = [1,3]
        test_env.start = np.array([5.0,5.0])
        test_env.goal = np.array([45.0,45.0])

        test_env.robot.N = 5

        test_env.num_cores = n_cores
        test_env.num_obs = n_obs

        observations.append(test_env.reset())

    return observations

def run_experiment(n_obs,n_cores):
    num = 500
    agents = [IQN_agent_0,IQN_agent_1,IQN_agent_2,IQN_agent_3,IQN_agent_4,DQN_agent_1,APF_agent,BA_agent]
    names = ["adaptive_IQN","IQN_0.25","IQN_0.5","IQN_0.75","IQN_1.0","DQN","APF","BA"]
    envs = [test_env_0,test_env_1,test_env_2,test_env_3,test_env_4,test_env_5,test_env_6,test_env_7]
    evaluations = [evaluation_IQN,evaluation_IQN,evaluation_IQN,evaluation_IQN,evaluation_IQN,evaluation_DQN, \
                   evaluation_classical,evaluation_classical]

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    exp_data = {}
    for name in names:
        exp_data[name] = dict(ep_data=[],success=[],time=[],energy=[],out_of_area=[],computation_times=[])

    print(f"Running {num} experiments\n")
    for i in range(num):
        observations = exp_setup_5(envs,n_obs,n_cores)
        for j in range(len(agents)):
            agent = agents[j]
            env = envs[j]
            evaluation = evaluations[j]
            name = names[j]
            
            # obs = env.reset()
            obs = observations[j]
            
            if name == "adaptive_IQN":
                ep_data, success, time, energy, out_of_area, computation_times = evaluation(obs,agent,env,adaptive=True)
            elif name == "IQN_0.25":
                ep_data, success, time, energy, out_of_area, computation_times = evaluation(obs,agent,env,cvar=0.25)
            elif name == "IQN_0.5":
                ep_data, success, time, energy, out_of_area, computation_times = evaluation(obs,agent,env,cvar=0.5)
            elif name == "IQN_0.75":
                ep_data, success, time, energy, out_of_area, computation_times = evaluation(obs,agent,env,cvar=0.75)
            else:
                ep_data, success, time, energy, out_of_area, computation_times = evaluation(obs,agent,env)
            
            exp_data[name]["ep_data"].append(ep_data)
            exp_data[name]["success"].append(success)
            exp_data[name]["time"].append(time)
            exp_data[name]["energy"].append(energy)
            exp_data[name]["out_of_area"].append(out_of_area)
            for compute_t in computation_times:
                 exp_data[name]["computation_times"].append(compute_t)

        if (i+1) % 10 == 0:
            print(f"=== Finish {i+1} experiments ===")

            for k in range(len(agents)):
                name = names[k]
                res = np.array(exp_data[name]["success"])
                idx = np.where(res == 1)[0]
                s_rate = np.sum(res)/(i+1)
                o_rate = np.sum(exp_data[name]["out_of_area"])/(i+1)
                
                t = np.array(exp_data[name]["time"])
                e = np.array(exp_data[name]["energy"])
                avg_t = np.mean(t[idx])
                avg_e = np.mean(e[idx])

                avg_comput_t = np.mean(exp_data[name]["computation_times"])
                
                print(f"{name} | success rate: {s_rate:.2f} | out of area rate: {o_rate:.2f} | avg_time: {avg_t:.2f} | avg_energy: {avg_e:.2f} | avg_compute_t: {avg_comput_t}")
            
            print("\n")

            filename = f"experiment_data/exp_data_{timestamp}.json"
            with open(filename,"w") as file:
                json.dump(exp_data,file)

if __name__ == "__main__":
    seed = 15 # PRNG seed for all testing envs

    ##### adaptive IQN #####
    test_env_0 = marinenav_env.MarineNavEnv(seed)

    save_dir = "training_data/training_2023-02-08-00-06-53/seed_3"

    # device = "cuda:0"
    device = "cpu"

    IQN_agent_0 = IQNAgent(test_env_0.get_state_space_dimension(),
                         test_env_0.get_action_space_dimension(),
                         device=device,
                         seed=2)
    IQN_agent_0.load_model(save_dir,device)
    ##### adaptive IQN #####


    ##### IQN cvar = 0.25 #####
    test_env_1 = marinenav_env.MarineNavEnv(seed)

    save_dir = "training_data/training_2023-02-08-00-06-53/seed_3"

    # device = "cuda:0"
    device = "cpu"

    IQN_agent_1 = IQNAgent(test_env_1.get_state_space_dimension(),
                         test_env_1.get_action_space_dimension(),
                         device=device,
                         seed=2)
    IQN_agent_1.load_model(save_dir,device)
    ##### IQN cvar = 0.25 #####


    ##### IQN cvar = 0.5 #####
    test_env_2 = marinenav_env.MarineNavEnv(seed)

    save_dir = "training_data/training_2023-02-08-00-06-53/seed_3"

    # device = "cuda:0"
    device = "cpu"

    IQN_agent_2 = IQNAgent(test_env_2.get_state_space_dimension(),
                         test_env_2.get_action_space_dimension(),
                         device=device,
                         seed=2)
    IQN_agent_2.load_model(save_dir,device)
    ##### IQN cvar = 0.5 #####


    ##### IQN cvar = 0.75 #####
    test_env_3 = marinenav_env.MarineNavEnv(seed)

    save_dir = "training_data/training_2023-02-08-00-06-53/seed_3"

    # device = "cuda:0"
    device = "cpu"

    IQN_agent_3 = IQNAgent(test_env_3.get_state_space_dimension(),
                         test_env_3.get_action_space_dimension(),
                         device=device,
                         seed=2)
    IQN_agent_3.load_model(save_dir,device)
    ##### IQN cvar = 0.75 #####
    

    ##### IQN cvar = 1.0 (greedy) #####
    test_env_4 = marinenav_env.MarineNavEnv(seed)

    save_dir = "training_data/training_2023-02-08-00-06-53/seed_3"

    # device = "cuda:0"
    device = "cpu"

    IQN_agent_4 = IQNAgent(test_env_4.get_state_space_dimension(),
                         test_env_4.get_action_space_dimension(),
                         device=device,
                         seed=2)
    IQN_agent_4.load_model(save_dir,device)
    ##### IQN cvar = 1.0 (greedy) #####


    ##### DQN #####
    test_env_5 = marinenav_env.MarineNavEnv(seed)
    
    save_dir = "training_data/training_2023-02-08-00-13-06/seed_3"
    model_file = "latest_model.zip"

    # device = "cuda:0"
    device = "cpu"
    DQN_agent_1 = DQN.load(os.path.join(save_dir,model_file),print_system_info=False,device=device)
    ##### DQN #####


    ##### APF #####
    test_env_6 = marinenav_env.MarineNavEnv(seed)
    
    APF_agent = APF.APF_agent(test_env_6.robot.a,test_env_6.robot.w)
    ##### APF #####


    ##### BA #####
    test_env_7 = marinenav_env.MarineNavEnv(seed)
    
    BA_agent = BA.BA_agent(test_env_7.robot.a,test_env_7.robot.w)
    ##### BA #####

    for n_obs,n_cores in [[10,8],[6,4]]: 
        run_experiment(n_obs,n_cores)

