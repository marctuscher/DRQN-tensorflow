from src.env_wrapper import GymWrapper, RetroWrapper
import numpy as np

class BaseAgent():

    def __init__(self, config):
        self.config = config
        if config.state != None:
            self.env_wrapper = RetroWrapper(config)
        else:
            self.env_wrapper = GymWrapper(config)
        self.rewards = 0
        self.lens = 0
        self.epsilon = config.epsilon_start
        self.min_reward = -1.
        self.max_reward = 1.0
        self.replay_memory = None
        self.history = None
        self.net = None
        if self.config.restore:
            self.load()
        else:
            self.i = 0



    def save(self):
        self.replay_memory.save()
        self.net.save_session()
        np.save(self.config.dir_save+'step.npy', self.i)

    def load(self):
        self.replay_memory.load()
        self.net.restore_session()
        self.i = np.load(self.config.dir_save+'step.npy')

