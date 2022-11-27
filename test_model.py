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
    
    positions = [c-16*d,c-8*d,c,c+8*d,c+16*d]

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

if __name__ == "__main__":

    save_dir = "experiment_dist_reward_energy_penalty"
    model_file = "best_model.zip"

    DQN_agent = DQN.load(os.path.join(save_dir,model_file),print_system_info=True)

    test_env = gym.make('marinenav_env:marinenav_env-v0')

    ev = env_visualizer.EnvVisualizer()

    first_obs = reset_scenario_1()

    evaluation(first_obs,DQN_agent)

    ev.load_episode(test_env.episode_data())

    ev.play_episode()

