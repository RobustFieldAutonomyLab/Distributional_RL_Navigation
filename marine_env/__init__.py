from gym.envs.registration import register

register(
    id='marine_env-v0',
    entry_point='marine_env.envs:Env',
)