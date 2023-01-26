import sys
sys.path.insert(0,"../")
import env_visualizer
import json
import numpy as np

if __name__ == "__main__":
    # filename = "../experiment_data/exp_data_1.json"
    # filename = "../experiment_data/exp_data_2023-01-25-16-18-30.json"
    # filename = "../experiment_data/exp_data_2023-01-25-17-28-28.json"
    filename = "../experiment_data/exp_data_2023-01-25-19-06-28.json"

    with open(filename,"r") as f:
        exp_data = json.load(f)

    names = ["IQN_agent_1","DQN_agent_1","APF_agent","BA_agent"]
    for k in range(len(names)):
        name = names[k]
        res = np.array(exp_data[name]["success"])
        idx = np.where(res == 1)[0]
        rate = np.sum(res)/np.shape(res)[0]
        
        t = np.array(exp_data[name]["time"])
        e = np.array(exp_data[name]["energy"])
        avg_t = np.mean(t[idx])
        avg_e = np.mean(e[idx])
        print(f"{name} | success rate: {rate:.2f} | avg_time: {avg_t:.2f} | avg_energy: {avg_e:.2f}")
    
    print("\n")
    
    agent = "APF_agent"
    
    # ep_id = 257 # strong adverse current flows
    ep_id = 384 # strong adverse current flows

    ev = env_visualizer.EnvVisualizer()

    episode = exp_data[agent]["ep_data"][ep_id]
    ev.load_episode(episode)
    ev.play_episode()
    