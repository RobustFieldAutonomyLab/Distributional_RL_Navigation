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

def evaluation_IQN(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0
    
    quantiles_data = []
    taus_data = []

    cvars = [1.0]

    while not done and length < 1000:
        action = None
        select = 0
        for i,cvar in enumerate(cvars):
            a, quantiles, taus = agent.act_eval_IQN(observation,0.0,cvar)
            if i == select:
                action = a

            if len(quantiles_data) < len(cvars):
                quantiles_data.append(quantiles)
                taus_data.append(taus)
            else:
                quantiles_data[i] = np.concatenate((quantiles_data[i],quantiles))
                taus_data[i] = np.concatenate((taus_data[i],taus))
        
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))
        
    # metric data
    success = True if info["state"] == "reach goal" else False
    time = test_env.robot.dt * test_env.robot.N * length

    ep_data = test_env.episode_data()
    ep_data["robot"]["actions_cvars"] = copy.deepcopy(cvars)
    ep_data["robot"]["actions_quantiles"] = [x.tolist() for x in quantiles_data]
    ep_data["robot"]["actions_taus"] = [x.tolist() for x in taus_data]

    return ep_data, success, time, energy

def evaluation_DQN(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0

    while not done and length < 1000:
        action, _ = agent.predict(observation,deterministic=True)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))
        
    # metric data
    success = True if info["state"] == "reach goal" else False
    time = test_env.robot.dt * test_env.robot.N * length

    return test_env.episode_data(), success, time, energy

def evaluation_classical(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0
    
    while not done and length < 1000:
        action = agent.act(observation)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))

    # metric data
    success = True if info["state"] == "reach goal" else False
    time = test_env.robot.dt * test_env.robot.N * length

    return test_env.episode_data(), success, time, energy

def experiment_1():
    num = 500
    agents = [IQN_agent_1,DQN_agent_1,APF_agent,BA_agent]
    names = ["IQN_agent_1","DQN_agent_1","APF_agent","BA_agent"]
    envs = [test_env_1,test_env_3,test_env_5,test_env_6]
    evaluations = [evaluation_IQN,evaluation_DQN, \
                   evaluation_classical,evaluation_classical]

    exp_data = {}
    for name in names:
        exp_data[name] = dict(ep_data=[],success=[],time=[],energy=[])

    print(f"Running {num} experiments\n")
    for i in range(num):
        for j in range(len(agents)):
            agent = agents[j]
            env = envs[j]
            evaluation = evaluations[j]
            name = names[j]
            
            obs = env.reset()
            ep_data, success, time, energy = evaluation(obs,agent,env)
            exp_data[name]["ep_data"].append(ep_data)
            exp_data[name]["success"].append(success)
            exp_data[name]["time"].append(time)
            exp_data[name]["energy"].append(energy)


        if (i+1) % 10 == 0:
            print(f"=== Finish {i+1} experiments ===")

            for k in range(len(agents)):
                name = names[k]
                rate = np.sum(exp_data[name]["success"])/(i+1)
                avg_t = np.mean(exp_data[name]["time"])
                avg_e = np.mean(exp_data[name]["energy"])
                print(f"{name} | success rate: {rate:.2f} | avg_time: {avg_t:.2f} | avg_energy: {avg_e:.2f}")
            
            print("\n")

            filename = "exp_data.json"
            with open(filename,"w") as file:
                json.dump(exp_data,file)

    

if __name__ == "__main__":
    seed = 15 # PRNG seed for all testing envs
    
    ##### IQN #####
    test_env_1 = marinenav_env.MarineNavEnv(seed)

    save_dir = "experiment_data/experiment_2022-12-23-18-02-05/seed_2"

    device = "cuda:0"

    IQN_agent_1 = IQNAgent(test_env_1.get_state_space_dimension(),
                         test_env_1.get_action_space_dimension(),
                         device=device,
                         seed=2)
    IQN_agent_1.load_model(save_dir,device)
    ##### IQN #####


    # ##### IQN with angle penalty #####
    # test_env_2 = marinenav_env.MarineNavEnv(seed)

    # save_dir = "experiment_data/experiment_2023-01-19-22-58-37/seed_2"

    # device = "cuda:0"

    # IQN_agent_2 = IQNAgent(test_env_2.get_state_space_dimension(),
    #                      test_env_2.get_action_space_dimension(),
    #                      device=device,
    #                      seed=2)
    # IQN_agent_2.load_model(save_dir,device)
    # ##### IQN with angle penalty #####


    ##### DQN #####
    test_env_3 = marinenav_env.MarineNavEnv(seed)
    
    save_dir = "experiment_data/experiment_2022-12-23-18-19-03/seed_2"
    model_file = "latest_model.zip"

    DQN_agent_1 = DQN.load(os.path.join(save_dir,model_file),print_system_info=False)
    ##### DQN #####


    # ##### DQN with angle penalty #####
    # test_env_4 = marinenav_env.MarineNavEnv(seed)

    # save_dir = "experiment_data/experiment_2023-01-20-19-56-05/seed_2"
    # model_file = "latest_model.zip"

    # DQN_agent_2 = DQN.load(os.path.join(save_dir,model_file),print_system_info=False)
    # ##### DQN with angle penalty #####


    ##### APF #####
    test_env_5 = marinenav_env.MarineNavEnv(seed)
    
    APF_agent = APF.APF_agent(test_env_5.robot.a,test_env_5.robot.w)

    ##### APF #####


    ##### BA #####
    test_env_6 = marinenav_env.MarineNavEnv(seed)
    
    BA_agent = BA.BA_agent(test_env_6.robot.a,test_env_6.robot.w)
    
    ##### BA #####

    experiment_1()

