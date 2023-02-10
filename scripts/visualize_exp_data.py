import sys
sys.path.insert(0,"../")
import env_visualizer
import json
import numpy as np

if __name__ == "__main__":
    # filename = "../experiment_data/exp_data_demonstration.json"
    filename = "../experiment_data/exp_data_2023-02-09-00-20-42.json"

    with open(filename,"r") as f:
        exp_data = json.load(f)

    names = ["adaptive_IQN","IQN_0.25","IQN_0.5","IQN_0.75","IQN_1.0","DQN","APF","BA"]

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
        # add = True
        # for name in names:
        #     if not exp_data[name]["success"][i]:
        #         add = False
        #         break
        # if add:
        #     ep_list.append(i)

        if exp_data["adaptive_IQN"]["success"][i] and not exp_data["IQN_1.0"]["success"][i]:
            ep_list.append(i)
    
    print(len(ep_list))
    print(ep_list)

    # ep_id = 110
    ep_id = 67

    plot_agents = ["adaptive_IQN","IQN_1.0"]
    action_sequences = {}

    print(f"Episode {ep_id} results:")
    min_len = np.inf
    for name in plot_agents:
        res = exp_data[name]["success"][ep_id]
        t = np.array(exp_data[name]["time"][ep_id])
        e = np.array(exp_data[name]["energy"][ep_id])
        action_sequences[name] = exp_data[name]["ep_data"][ep_id]["robot"]["action_history"]
        min_len = min(min_len,len(action_sequences[name]))
        print(f"{name}| success: {res} | time: {t} | energy: {e}")

    # identify the fork state where adaptive IQN and IQN choose different actions for the first time
    fork_state = None
    for i in range(min_len):
        if action_sequences[plot_agents[0]][i] != action_sequences[plot_agents[1]][i]:
            fork_state = {}
            fork_state["id"] = i
            fork_state["cvars"] = [exp_data[plot_agents[0]]["ep_data"][ep_id]["robot"]["actions_cvars"][i],
                                   exp_data[plot_agents[1]]["ep_data"][ep_id]["robot"]["actions_cvars"][i]]
            fork_state["quantiles"] = [exp_data[plot_agents[0]]["ep_data"][ep_id]["robot"]["actions_quantiles"][i][0],
                                       exp_data[plot_agents[1]]["ep_data"][ep_id]["robot"]["actions_quantiles"][i][0]]
            break

    ev = env_visualizer.EnvVisualizer(draw_dist=True,cvar_num=2)

    episode = exp_data[plot_agents[1]]["ep_data"][ep_id]
    ev.load_episode(episode)

    # ev.play_episode()
    
    # Draw trajectorys
    ev.draw_trajectory(only_ep_actions=False,all_actions=action_sequences,fork_state_info=fork_state)

    ev.fig.tight_layout()
    ev.fig.savefig("cvar_distributions.png")
    