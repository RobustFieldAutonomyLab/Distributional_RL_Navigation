import os
import sys
sys.path.insert(0,"../")
sys.path.insert(0,"../thirdparty")
from thirdparty import IQNAgent,DQN
import json
import APF
import BA
import marinenav_env.envs.marinenav_env as marinenav_env
import env_visualizer
import copy

def evaluation_IQN(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0
    
    quantiles_data = []
    taus_data = []

    cvars = []

    while not done and length < 1000:
        action = None

        (action, quantiles_0, taus_0), cvar = agent.act_adaptive_eval(observation)
        _, quantiles_1, taus_1 = agent.act_eval(observation,cvar=1.0)
        
        cvars.append([cvar,1.0])
        quantiles_data.append([quantiles_0.tolist()[0],quantiles_1.tolist()[0]])
        taus_data.append([taus_0.tolist(),taus_1.tolist()])
        
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
    ep_data["robot"]["actions_quantiles"] = copy.deepcopy(quantiles_data)
    ep_data["robot"]["actions_taus"] = copy.deepcopy(taus_data)

    return ep_data, success, time, energy, out_of_area

def evaluation_DQN(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0

    q_value_data = []

    while not done and length < 1000:
        # TODO: Q value are not output by DQN in the current code 
        q_values,action, _ = agent.predict(observation,deterministic=True)
        q_value_data.append(q_values.tolist()[0])
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))
        
    # metric data
    success = True if info["state"] == "reach goal" else False
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length
    ep_data = test_env.episode_data()
    ep_data["robot"]["actions_values"] = copy.deepcopy(q_value_data)

    return ep_data, success, time, energy, out_of_area

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
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length

    return test_env.episode_data(), success, time, energy, out_of_area

config_name = "video_config_2.json"
with open(config_name,"r") as f:
    config = json.load(f)

agents = ["adaptive_IQN","DQN","APF","BA"]

for agent in agents:
    seed = 15
    test_env = marinenav_env.MarineNavEnv(seed)
    observation = test_env.reset_with_eval_config(config)

    if agent == "adaptive_IQN":
        ##### adaptive IQN #####
        save_dir = "../training_data/training_2023-02-08-00-06-53/seed_3"
        device = "cpu"

        IQN_agent = IQNAgent(test_env.get_state_space_dimension(),
                            test_env.get_action_space_dimension(),
                            device=device,
                            seed=2)
        IQN_agent.load_model(save_dir,device)
        episode, success, time, energy, out_of_area = evaluation_IQN(observation,IQN_agent,test_env)
        ##### adaptive IQN #####
    elif agent == "DQN":
        ##### DQN #####
        save_dir = "../training_data/training_2023-02-08-00-13-06/seed_3"
        model_file = "latest_model.zip"

        device = "cpu"
        DQN_agent = DQN.load(os.path.join(save_dir,model_file),print_system_info=False,device=device)
        episode, success, time, energy, out_of_area = evaluation_DQN(observation,DQN_agent,test_env)
        ##### DQN #####
    elif agent == "APF":
        ##### APF #####
        APF_agent = APF.APF_agent(test_env.robot.a,test_env.robot.w)

        episode, success, time, energy, out_of_area = evaluation_classical(observation,APF_agent,test_env)
        ##### APF #####
    elif agent == "BA":
        ##### BA #####
        BA_agent = BA.BA_agent(test_env.robot.a,test_env.robot.w)

        episode, success, time, energy, out_of_area = evaluation_classical(observation,BA_agent,test_env)
        ##### BA #####


    with open(f"{agent}_video.json","w") as file:
        json.dump(episode,file)

    ev = env_visualizer.EnvVisualizer()
    ev.load_episode(episode)
    # Draw trajectorys
    ev.draw_trajectory()
    # ev.draw_trajectory(only_ep_actions=False,all_actions=action_sequences,fork_state_info=fork_state)

    ev.fig.tight_layout()
    ev.fig.savefig(f"{agent}_debug.png")
