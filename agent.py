class Agent:
    
    def __init__(self):
        pass

    def act(self, observation):
        return self.policy(observation)