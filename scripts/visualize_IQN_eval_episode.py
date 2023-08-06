import sys
sys.path.insert(0,"../")
import env_visualizer

ev = env_visualizer.EnvVisualizer()

config_file = "../pretrained_models/IQN/seed_3/eval_config.json"
eval_file = "../pretrained_models/IQN/seed_3/greedy_evaluations.npz"

# The index of evaluation 
eval_id = 99

# The index of episode in an evluation
env_id = 3

ev.load_episode_from_eval_files(config_file,eval_file,eval_id,env_id)

ev.play_episode()