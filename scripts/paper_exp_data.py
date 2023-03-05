import sys
sys.path.insert(0,"../")
import json
import numpy as np

if __name__ == "__main__":
    # filename = "../experiment_data/exp_data_demonstration.json"
    # test_1_file = "../experiment_data/exp_data_2023-02-21-15-53-30.json"
    # test_2_file = "../experiment_data/exp_data_2023-02-21-15-08-07.json"
    test_1_file = "../experiment_data/exp_data_test_case_1_cpu.json"
    test_2_file = "../experiment_data/exp_data_test_case_2_cpu.json"

    filenames = [test_1_file,test_2_file]

    names = ["adaptive_IQN","IQN_0.25","IQN_0.5","IQN_0.75","IQN_1.0","DQN","APF","BA"]
    compute_t = {}
    for name in names:
        compute_t[name] = []
    
    for filename in filenames:
        with open(filename,"r") as f:
            exp_data = json.load(f)

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

            for comp_t in exp_data[name]["computation_times"]:
                compute_t[name].append(comp_t)

            avg_compute_t = np.mean(exp_data[name]["computation_times"])
            std_compute_e = np.std(exp_data[name]["computation_times"])

            print(f"{name} | success rate: {s_rate:.2f} | out of area rate: {o_rate:.2f} | time: {avg_t:.2f} +- {std_t:.2f} | energy: {avg_e:.2f} +- {std_e:.2f} | compute_t: {avg_compute_t} +- {std_compute_e}")
        
        print("\n")

    for name in names:
        avg_compute_t = np.mean(compute_t[name])
        max_compute_e = np.max(compute_t[name])
        print(f"{name} | avg_t: {avg_compute_t} | max_t: {max_compute_e}")