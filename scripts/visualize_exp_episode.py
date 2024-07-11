import sys
sys.path.insert(0,"../")
import env_visualizer
import json
import numpy as np

filename = "your/experiment/result/file"

with open(filename,"r") as f:
    exp_data = json.load(f)

episode_id = 0
agent = "IQN"

ev = env_visualizer.EnvVisualizer()

episode = exp_data[agent]["ep_data"][episode_id]
ev.load_episode(episode)

ev.play_episode()