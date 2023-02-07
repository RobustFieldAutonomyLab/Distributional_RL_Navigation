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
import time

def evaluation(first_observation, agent):
    print("===== Evaluation =====")
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    
    while not done and length < 1000:
        action, _ = agent.predict(observation,deterministic=True)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        # if length % 50 == 0:
        #     print(length)

    res = "success!" if info["state"] == "reach goal" else "failed!" 
    print(res)
    print("episode length: ",length)
    print("cumulative reward: ",cumulative_reward)

    return test_env.episode_data()

def evaluation_IQN(first_observation, agent):
    print("===== Evaluate IQN =====")
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    
    quantiles_data = []
    taus_data = []

    cvars = [1.0,0.7,0.4,0.1]
    # cvars = [0.7]

    start = time.time()
    while not done and length < 1000:
        action = None
        select = 0
        for i,cvar in enumerate(cvars):
            a, quantiles, taus = agent.act_eval(observation,cvar=cvar)
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
        if length % 50 == 0:
            print(length)

    end = time.time()
    print("time: ",end-start)

    res = "success!" if info["state"] == "reach goal" else "failed!" 
    print(res)
    print("episode length: ",length)
    print("cumulative reward: ",cumulative_reward)

    ep_data = test_env.episode_data()
    ep_data["robot"]["actions_cvars"] = copy.deepcopy(cvars)
    ep_data["robot"]["actions_quantiles"] = [x.tolist() for x in quantiles_data]
    ep_data["robot"]["actions_taus"] = [x.tolist() for x in taus_data]

    return ep_data

def evaluation_classical(first_observation, agent):
    print("===== Evaluation =====")
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    
    while not done and length < 1000:
        action = agent.act(observation)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        # if length % 50 == 0:
        #     print(length)

    res = "success!" if info["state"] == "reach goal" else "failed!" 
    print(res)
    print("episode length: ",length)
    print("cumulative reward: ",cumulative_reward)

    return test_env.episode_data()

def reset_episode_scenario():
    current_v = test_env.get_velocity(test_env.start[0],test_env.start[1])
    test_env.robot.reset_state(test_env.start[0],test_env.start[1], current_velocity=current_v)
    return test_env.get_observation()

def reset_scenario_1():
    test_env.cores.clear()
    test_env.obstacles.clear()

    c = np.array([25.0,25.0])

    d1 = np.array([-1.0,1.0])
    d1 /= np.linalg.norm(d1)
    d2 = np.array([1.0,-1.0])
    d2 /= np.linalg.norm(d2)
    
    # set start and goal position
    test_env.start = np.array([10.0,10.0])
    test_env.goal = np.array([40.0,40.0])
    
    positions = [c+12*d1,c+6*d1,c,c+6*d2,c+12*d2]
    # positions = list(np.linspace(c-12*d,c+12*d,5))

    # set a vortex core
    core_1 = marinenav_env.Core(15.0,19.0,0,np.pi*10)
    test_env.cores.append(core_1)
    core_2 = marinenav_env.Core(19.0,15.0,1,np.pi*10)
    test_env.cores.append(core_2)
    c_centers = np.array([[core_1.x,core_1.y],[core_2.x,core_2.y]])
    test_env.core_centers = scipy.spatial.KDTree(c_centers)

    centers = None
    for position in positions:
        obs = marinenav_env.Obstacle(position[0],position[1],3.0)
        test_env.obstacles.append(obs)
        if centers is None:
            centers = np.array([[obs.x,obs.y]])
        else:
            c = np.array([[obs.x,obs.y]])
            centers = np.vstack((centers,c))

    test_env.obs_centers = scipy.spatial.KDTree(centers)

    # reset robot
    test_env.robot.init_theta = np.pi/4
    test_env.robot.init_speed = 0.0
    current_v = test_env.get_velocity(test_env.start[0],test_env.start[1])
    test_env.robot.reset_state(test_env.start[0],test_env.start[1], current_velocity=current_v)

    return test_env.get_observation()

def reset_scenario_2():
    test_env.cores.clear()
    test_env.obstacles.clear()

    # set start and goal position
    test_env.start = np.array([6.0,6.0])
    test_env.goal = np.array([44.0,44.0])

    # set a vortex core
    core_1 = marinenav_env.Core(15.0,19.0,0,np.pi*10)
    test_env.cores.append(core_1)
    core_2 = marinenav_env.Core(19.0,15.0,1,np.pi*10)
    test_env.cores.append(core_2)
    c_centers = np.array([[core_1.x,core_1.y],[core_2.x,core_2.y]])
    test_env.core_centers = scipy.spatial.KDTree(c_centers)

    # obstacle
    c = np.array([27.0,27.0])
    obs = marinenav_env.Obstacle(c[0],c[1],10.0)
    test_env.obstacles.append(obs)
    centers = np.array([[obs.x,obs.y]])
    test_env.obs_centers = scipy.spatial.KDTree(centers)

    # reset robot
    test_env.robot.init_theta = np.pi/4
    test_env.robot.init_speed = 0.0
    current_v = test_env.get_velocity(test_env.start[0],test_env.start[1])
    test_env.robot.reset_state(test_env.start[0],test_env.start[1], current_velocity=current_v)

    return test_env.get_observation()

def reset_scenario_3():
    test_env.cores.clear()
    test_env.obstacles.clear()

    # set start and goal position
    test_env.start = np.array([20.0,15.0])
    test_env.goal = np.array([25.0,35.0])

    # obstacle
    c = np.array([25.0,25.0])
    obs = marinenav_env.Obstacle(c[0],c[1],5.0)
    test_env.obstacles.append(obs)
    centers = np.array([[obs.x,obs.y]])
    test_env.obs_centers = scipy.spatial.KDTree(centers)

    # set a strong vortex core near the obstacle
    d = np.array([1.0,-1.0])
    d /= np.linalg.norm(d)
    pos = c + 4 * d
    core = marinenav_env.Core(pos[0],pos[1],0,2*np.pi*test_env.r * 10)
    test_env.cores.append(core)
    c_centers = np.array([[core.x,core.y]])
    test_env.core_centers = scipy.spatial.KDTree(c_centers)

    # reset robot
    test_env.robot.init_theta = 0.0
    test_env.robot.init_speed = 0.0
    current_v = test_env.get_velocity(test_env.start[0],test_env.start[1])
    test_env.robot.reset_state(test_env.start[0],test_env.start[1], current_velocity=current_v)

    return test_env.get_observation()

def reset_scenario_4():
    test_env.cores.clear()
    test_env.obstacles.clear()

    # set start and goal position
    test_env.start = np.array([37.5,24.0])
    test_env.goal = np.array([12.5,33.0])

    # obstacle
    obs_1 = marinenav_env.Obstacle(27.0,32.0,2.0)
    test_env.obstacles.append(obs_1)
    obs_2 = marinenav_env.Obstacle(32.5,35.0,1.0)
    test_env.obstacles.append(obs_2)
    centers = np.array([[obs_1.x,obs_1.y],[obs_2.x,obs_2.y]])
    test_env.obs_centers = scipy.spatial.KDTree(centers)

    # set a vortex core
    core = marinenav_env.Core(31.0,30.0,0,np.pi*5)
    test_env.cores.append(core)
    c_centers = np.array([[core.x,core.y]])
    test_env.core_centers = scipy.spatial.KDTree(c_centers)

    # reset robot
    test_env.robot.init_theta = np.pi/2
    test_env.robot.init_speed = 1.0
    current_v = test_env.get_velocity(test_env.start[0],test_env.start[1])
    test_env.robot.reset_state(test_env.start[0],test_env.start[1], current_velocity=current_v)

    return test_env.get_observation()

def reset_random():
    # reset the environment
        
    test_env.cores.clear()
    test_env.obstacles.clear()

    num_cores = 1
    num_obs = 1

    # reset start and goal state randomly
    iteration = 500
    max_dist = 0.0
    while True:
        start = test_env.rd.uniform(low = 5.0*np.ones(2), high = np.array([test_env.width-5.0,test_env.height-5.0]))
        goal = test_env.rd.uniform(low = 5.0*np.ones(2), high = np.array([test_env.width-5.0,test_env.height-5.0]))
        iteration -= 1
        if np.linalg.norm(goal-start) > max_dist:
            max_dist = np.linalg.norm(goal-start)
            test_env.start = start
            test_env.goal = goal
        if max_dist > 25.0 or iteration == 0:
            break

    # generate vortex with random position, spinning direction and strength
    if num_cores > 0:
        iteration = 500
        while True:
            center = test_env.rd.uniform(low = np.zeros(2), high = np.array([test_env.width,test_env.height]))
            direction = test_env.rd.binomial(1,0.5)
            v_edge = test_env.rd.uniform(low = test_env.v_range[0], high = test_env.v_range[1])
            Gamma = 2 * np.pi * test_env.r * v_edge
            core = marinenav_env.Core(center[0],center[1],direction,Gamma)
            iteration -= 1
            if test_env.check_core(core):
                test_env.cores.append(core)
                num_cores -= 1
            if iteration == 0 or num_cores == 0:
                break
    
    centers = None
    for core in test_env.cores:
        if centers is None:
            centers = np.array([[core.x,core.y]])
        else:
            c = np.array([[core.x,core.y]])
            centers = np.vstack((centers,c))
    
    # KDTree storing vortex core center positions
    if centers is not None:
        test_env.core_centers = scipy.spatial.KDTree(centers)

    # generate obstacles with random position and size
    if num_obs > 0:
        iteration = 500
        while True:
            center = test_env.rd.uniform(low = np.zeros(2), high = np.array([test_env.width,test_env.height]))
            r = test_env.rd.uniform(low = test_env.obs_r_range[0], high = test_env.obs_r_range[1])
            obs = marinenav_env.Obstacle(center[0],center[1],r)
            iteration -= 1
            if test_env.check_obstacle(obs):
                test_env.obstacles.append(obs)
                num_obs -= 1
            if iteration == 0 or num_obs == 0:
                break

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

    # reset robot state
    test_env.reset_robot()

    return test_env.get_observation()

def reset_scenario_test_APF():
    # reset the environment
        
    test_env.cores.clear()
    test_env.obstacles.clear()

    # set start and goal position
    test_env.start = np.array([10.0,10.0])
    test_env.goal = np.array([44.0,44.0])

    # obstacle
    c = np.array([25.0,25.0])
    obs = marinenav_env.Obstacle(c[0],c[1],10.0)
    test_env.obstacles.append(obs)
    centers = np.array([[obs.x,obs.y]])
    test_env.obs_centers = scipy.spatial.KDTree(centers)

    # reset robot
    test_env.robot.init_theta = 0.0
    test_env.robot.init_speed = 0.0
    current_v = test_env.get_velocity(test_env.start[0],test_env.start[1])
    test_env.robot.reset_state(test_env.start[0],test_env.start[1], current_velocity=current_v)

    return test_env.get_observation()

def reset_scenario_strong_adverse_current():
    test_env.cores.clear()
    test_env.obstacles.clear()
    
    # set start and goal
    test_env.start = np.array([15.0,10.0])
    test_env.goal = np.array([45.0,35.0])

    # set vortex cores
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

    return test_env.get_observation()

def reset_hard_test_env():
    test_env.cores.clear()
    test_env.obstacles.clear()
    
    # set start and goal
    test_env.start = np.array([5.0,5.0])
    test_env.goal = np.array([45.0,45.0])

    num_cores = 8
    num_obs = 10

    # generate vortex with random position, spinning direction and strength
    if num_cores > 0:
        iteration = 500
        while True:
            center = test_env.rd.uniform(low = np.zeros(2), high = np.array([test_env.width,test_env.height]))
            direction = test_env.rd.binomial(1,0.5)
            v_edge = test_env.rd.uniform(low = test_env.v_range[0], high = test_env.v_range[1])
            Gamma = 2 * np.pi * test_env.r * v_edge
            core = marinenav_env.Core(center[0],center[1],direction,Gamma)
            iteration -= 1
            if test_env.check_core(core):
                test_env.cores.append(core)
                num_cores -= 1
            if iteration == 0 or num_cores == 0:
                break
    
    centers = None
    for core in test_env.cores:
        if centers is None:
            centers = np.array([[core.x,core.y]])
        else:
            c = np.array([[core.x,core.y]])
            centers = np.vstack((centers,c))
    
    # KDTree storing vortex core center positions
    if centers is not None:
        test_env.core_centers = scipy.spatial.KDTree(centers)

    # generate obstacles with random position
    if num_obs > 0:
        iteration = 500
        while True:
            center = test_env.rd.uniform(low = 10*np.ones(2), high = 40*np.ones(2))
            r = 1.0
            obs = marinenav_env.Obstacle(center[0],center[1],r)
            iteration -= 1
            if test_env.check_obstacle(obs):
                test_env.obstacles.append(obs)
                num_obs -= 1
            if iteration == 0 or num_obs == 0:
                break

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

    # reset robot state
    test_env.reset_robot()

    return test_env.get_observation()

if __name__ == "__main__":

    # save_dir = "training_data/experiment_2022-12-23-18-02-05/seed_2" # IQN
    # save_dir = "training_data/training_2023-02-02-15-09-50/seed_2" # IQN
    save_dir = "training_data/training_2023-02-02-17-20-39/seed_2" # DQN
    # save_dir = "training_data/experiment_2023-01-19-22-58-47/seed_2" # QR-DQN with angle penalty
    model_file = "latest_model.zip"
    eval_file = "evaluations.npz"

    ev = env_visualizer.EnvVisualizer(seed=40)

    test_env = ev.env

    first_obs = reset_hard_test_env()

    ##### DQN #####
    DQN_agent = DQN.load(os.path.join(save_dir,model_file),print_system_info=True)

    ep_data = evaluation(first_obs,DQN_agent)
    ##### DQN #####

    ##### IQN #####
    # device = "cuda:0"

    # IQN_agent = IQNAgent(test_env.get_state_space_dimension(),
    #                      test_env.get_action_space_dimension(),
    #                      device=device,
    #                      seed=2)
    # IQN_agent.load_model(save_dir,device)

    # ep_data = evaluation_IQN(first_obs,IQN_agent)
    ##### IQN #####

    ##### QR-DQN #####
    # QRDQN_agent = QRDQN.load(os.path.join(save_dir,model_file),print_system_info=True)

    # ep_data = evaluation(first_obs,QRDQN_agent)
    ##### QR-DQN #####

    ##### APF #####
    # APF_agent = APF.APF_agent(test_env.robot.a,test_env.robot.w)
    
    # ep_data = evaluation_classical(first_obs,APF_agent)
    ##### APF #####

    ##### BA #####
    # BA_agent = BA.BA_agent(test_env.robot.a,test_env.robot.w)
    
    # ep_data = evaluation_classical(first_obs,BA_agent)
    ##### BA #####

    filename = "test.json"
    with open(filename,"w") as file:
        json.dump(ep_data,file)

    # test_env.save_episode("test.json")

    # ev_2 = env_visualizer.EnvVisualizer(draw_dist=True, cvar_num=4) # for IQN only
    ev_2 = env_visualizer.EnvVisualizer()
    
    ev_2.load_episode_from_json_file("test.json")

    ev_2.play_episode()

    # Draw trajectorys
    # ev_2.draw_trajectory()

