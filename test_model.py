import sys
sys.path.insert(0,"./thirdparty")
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from thirdparty import QRDQN
from thirdparty import IQNAgent
import os
import gym
import marinenav_env.envs.marinenav_env as marinenav_env
import numpy as np
import copy
import scipy.spatial
import env_visualizer

def evaluation(first_observation, agent):
    print("===== Evaluation =====")
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    
    while not done and length < 1000:
        action, _ = agent.predict(observation,deterministic=True)
        observation, reward, done, _ = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        if length % 50 == 0:
            print(length)

    print("episode length: ",length)
    print("cumulative reward: ",cumulative_reward)

def evaluation_IQN(first_observation, agent):
    print("===== Evaluate IQN =====")
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    
    while not done and length < 1000:
        action = agent.act(observation,0.0)
        observation, reward, done, _ = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        if length % 50 == 0:
            print(length)

    print("episode length: ",length)
    print("cumulative reward: ",cumulative_reward)

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

    # set a virtual vortex core
    core = marinenav_env.Core(50.5,50.5,0,0.0)
    test_env.cores.append(core)
    c_centers = np.array([[core.x,core.y]])
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
    

if __name__ == "__main__":

    save_dir = "experiment_data/experiment_2022-12-23-18-19-03/seed_6"
    model_file = "latest_model.zip"
    eval_file = "evaluations.npz"

    ev = env_visualizer.EnvVisualizer(seed=20)

    test_env = ev.env

    first_obs = reset_scenario_1()

    ##### DQN #####
    #DQN_agent = DQN.load(os.path.join(save_dir,model_file),print_system_info=True)

    #evaluation(first_obs,DQN_agent)
    ##### DQN #####

    ##### IQN #####
    # device = "cuda:0"

    # IQN_agent = IQNAgent(test_env.get_state_space_dimension(),
    #                      test_env.get_action_space_dimension(),
    #                      device=device,
    #                      seed=2)
    # IQN_agent.load_model(save_dir,device)

    # evaluation_IQN(first_obs,IQN_agent)
    ##### IQN #####

    ##### QR-DQN #####
    QRDQN_agent = QRDQN.load(os.path.join(save_dir,model_file),print_system_info=True)

    evaluation(first_obs,QRDQN_agent)
    ##### QR-DQN #####

    test_env.save_episode("test.json")

    ev_2 = env_visualizer.EnvVisualizer()
    
    ev_2.load_episode_from_json_file("test.json")

    ev_2.play_episode()

