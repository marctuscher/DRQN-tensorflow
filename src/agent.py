from src.env_wrapper import GymWrapper
import numpy as np

class BaseAgent():

    def __init__(self, config):
        self.config = config
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

    def play(self, episodes):
        self.env_wrapper.new_game()
        i = 0
        for _ in range(self.config.history_len):
            self.history.add(self.env_wrapper.screen)
        episode_steps = 0
        while i < episodes:
            a = self.net.q_action.eval({
                self.net.state : [self.history.get()],
                self.net.dropout: 1.0
            }, session=self.net.sess)
            self.env_wrapper.act_play(a[0])
            self.history.add(self.env_wrapper.screen)
            episode_steps += 1
            if episode_steps > self.config.max_steps:
                self.env_wrapper.terminal = True
            if self.env_wrapper.terminal:
                episode_steps = 0
                i += 1
                self.env_wrapper.new_game()
                for _ in range(self.config.history_len):
                    self.history.add(self.env_wrapper.screen)
