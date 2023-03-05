import os
import sys
sys.path.insert(0,"../")
import env_visualizer
import json

save_dir = "video"
os.makedirs(save_dir)
agents = ["adaptive_IQN"]

start_idx = 0
for agent in agents:
    filename = f"{agent}_video.json"

    with open(filename,"r") as f:
        episode = json.load(f)

    if agent == "adaptive_IQN":
        ev = env_visualizer.EnvVisualizer(video_plots=True,plot_dist=True,cvar_num=2)
    elif agent == "DQN":
        ev = env_visualizer.EnvVisualizer(video_plots=True,plot_qvalues=True)
    elif agent == "APF":
        episode["robot"]["action_history"][60:]=[]
        ev = env_visualizer.EnvVisualizer(video_plots=True)
    elif agent == "BA":
        episode["robot"]["action_history"][80:]=[]
        ev = env_visualizer.EnvVisualizer(video_plots=True)

    start_idx = ev.draw_video_plots(episode,save_dir,start_idx,agent)