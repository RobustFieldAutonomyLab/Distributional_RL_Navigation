import sys
sys.path.insert(0,"../")
import env_visualizer
import json
import numpy as np

if __name__ == "__main__":
    # filename = "../experiment_data/exp_data_1.json"
    # filename = "../experiment_data/exp_data_2023-01-25-16-18-30.json"
    # filename = "../experiment_data/exp_data_2023-01-25-17-28-28.json"
    # filename = "../experiment_data/exp_data_2023-01-25-19-06-28.json"
    # filename = "../experiment_data/exp_data_2023-01-26-21-32-02.json"
    filename = "../experiment_data/exp_data_2023-01-29-00-42-54.json"

    with open(filename,"r") as f:
        exp_data = json.load(f)

    names = ["adaptive_IQN","IQN","DQN","APF","BA"]

    for name in names:
        res = np.array(exp_data[name]["success"])
        idx = np.where(res == 1)[0]
        rate = np.sum(res)/np.shape(res)[0]
        
        t = np.array(exp_data[name]["time"])
        e = np.array(exp_data[name]["energy"])
        avg_t = np.mean(t[idx])
        avg_e = np.mean(e[idx])
        print(f"{name} | success rate: {rate:.2f} | avg_time: {avg_t:.2f} | avg_energy: {avg_e:.2f}")
    
    print("\n")


    ep_list = []
    for i in range(len(exp_data["adaptive_IQN"]["success"])):
        add = True
        for name in names:
            if not exp_data[name]["success"][i]:
                add = False
                break
        if add:
            ep_list.append(i)
        
    
    print(len(ep_list))
    print(ep_list)

    # ep_id = 259
    # ep_id = 374
    ep_id = 0

    plot_agents = ["adaptive_IQN","APF","BA"]
    action_sequences = {}

    print(f"Episode {ep_id} results:")
    for name in plot_agents:
        res = exp_data[name]["success"][ep_id]
        t = np.array(exp_data[name]["time"][ep_id])
        e = np.array(exp_data[name]["energy"][ep_id])
        action_sequences[name] = exp_data[name]["ep_data"][ep_id]["robot"]["action_history"]
        print(f"{name}| success: {res} | time: {t} | energy: {e}")

    ev = env_visualizer.EnvVisualizer(draw_traj=True)

    episode = exp_data["adaptive_IQN"]["ep_data"][ep_id]
    ev.load_episode(episode)
    
    # Draw trajectorys
    ev.draw_trajectory(only_ep_actions=False,all_actions=action_sequences)
    