import sys
sys.path.insert(0,"./thirdparty")
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
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

def reset_scenario_1():
    test_env.cores.clear()
    test_env.obstacles.clear()

    c = np.array([25.0,25.0])

    d = np.array([1.0,-1.0])
    d /= np.linalg.norm(d)
    
    
    # positions = [c-12*d,c-6*d,c,c+6*d,c+12*d]
    positions = list(np.linspace(c-13*d,c+13*d,5))

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

    test_env.reset_robot()

    return test_env.get_observation()

def reset_scenario_2():
    test_env.cores.clear()
    test_env.obstacles.clear()

    # set a virtual vortex core
    core = marinenav_env.Core(50.5,50.0,0,0.0)
    test_env.cores.append(core)
    c_centers = np.array([[core.x,core.y]])
    test_env.core_centers = scipy.spatial.KDTree(c_centers)

    c = np.array([25.0,25.0])
    obs = marinenav_env.Obstacle(c[0],c[1],20.0)
    test_env.obstacles.append(obs)
    centers = np.array([[obs.x,obs.y]])
    test_env.obs_centers = scipy.spatial.KDTree(centers)

    test_env.reset_robot()

    return test_env.get_observation()



if __name__ == "__main__":

    # save_dir = "experiment_dist_reward_energy_penalty"
    save_dir = "experiment_12_11"
    model_file = "latest_model.zip"
    eval_file = "evaluations.npz"

    DQN_agent = DQN.load(os.path.join(save_dir,model_file),print_system_info=True)

    ev = env_visualizer.EnvVisualizer()

    ev.load_episode_from_eval_file(os.path.join(save_dir,eval_file),-1)

    test_env = ev.env

    first_obs = reset_scenario_2()

    evaluation(first_obs,DQN_agent)

    test_env.save_episode("test.json")

    # ev.load_episode(test_env.episode_data())

    ev_2 = env_visualizer.EnvVisualizer()
    
    ev_2.load_episode_from_json_file("test.json")

    ev_2.play_episode()

