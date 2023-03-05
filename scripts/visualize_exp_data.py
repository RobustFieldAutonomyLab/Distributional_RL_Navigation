import sys
sys.path.insert(0,"../")
import env_visualizer
import json
import numpy as np

if __name__ == "__main__":
    # filename = "../experiment_data/exp_data_demonstration.json"
    # filename = "../experiment_data/exp_data_2023-02-21-15-08-07.json"
    # filename = "../experiment_data/exp_data_2023-02-26-18-08-41.json"
    filename = "../experiment_data/exp_data_test_case_2_cpu.json"

    with open(filename,"r") as f:
        exp_data = json.load(f)

    names = ["adaptive_IQN","IQN_0.25","IQN_0.5","IQN_0.75","IQN_1.0","DQN","APF","BA"]

    for name in names:
        res = np.array(exp_data[name]["success"])
        idx = np.where(res == 1)[0]
        s_rate = np.sum(res)/np.shape(res)[0]
        o_rate = np.sum(exp_data[name]["out_of_area"])/np.shape(res)[0]

        t = np.array(exp_data[name]["time"])
        e = np.array(exp_data[name]["energy"])
        avg_t = np.mean(t[idx])
        std_t = np.std(t[idx])
        avg_e = np.mean(e[idx])
        std_e = np.std(t[idx])

        avg_compute_t = np.mean(exp_data[name]["computation_times"])
        std_compute_e = np.std(exp_data[name]["computation_times"])

        print(f"{name} | success rate: {s_rate:.2f} | out of area rate: {o_rate:.2f} \
              | time: {avg_t:.2f} +- {std_t:.2f} | energy: {avg_e:.2f} +- {std_e:.2f} \
              | compute_t: {avg_compute_t} +- {std_compute_e}")
    
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

        if len(exp_data["APF"]["ep_data"][i]["robot"]["action_history"])>200:
        # if exp_data["adaptive_IQN"]["success"][i] and not exp_data["IQN_1.0"]["success"][i] and not exp_data["DQN"]["success"][i] \
        #    and not exp_data["APF"]["success"][i] and not exp_data["BA"]["success"][i]:
            ep_list.append(i)
    
    print(len(ep_list))
    print(ep_list)

    ep_id = 131
    # ep_id = 191

    plot_agents = ["adaptive_IQN","IQN_1.0","DQN","APF","BA"]
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

    # # identify the fork state where adaptive IQN and IQN choose different actions for the first time
    # fork_state = None
    # for i in range(min_len):
    #     if action_sequences[plot_agents[0]][i] != action_sequences[plot_agents[1]][i]:
    #         fork_state = {}
    #         fork_state["id"] = i
    #         fork_state["cvars"] = [exp_data[plot_agents[0]]["ep_data"][ep_id]["robot"]["actions_cvars"][i],
    #                                exp_data[plot_agents[1]]["ep_data"][ep_id]["robot"]["actions_cvars"][i]]
    #         fork_state["quantiles"] = [exp_data[plot_agents[0]]["ep_data"][ep_id]["robot"]["actions_quantiles"][i][0],
    #                                    exp_data[plot_agents[1]]["ep_data"][ep_id]["robot"]["actions_quantiles"][i][0]]
    #         break

    ev = env_visualizer.EnvVisualizer()

    episode = exp_data[plot_agents[1]]["ep_data"][ep_id]
    ev.load_episode(episode)

    # ev.play_episode()
    
    # Draw trajectorys
    ev.draw_trajectory(only_ep_actions=False,all_actions=action_sequences)
    # ev.draw_trajectory(only_ep_actions=False,all_actions=action_sequences,fork_state_info=fork_state)

    # ev.fig.tight_layout()
    # ev.fig.savefig("debug_trajectories.png")
    # ev.fig.savefig("cvar_distributions_cpu.png")
    