import numpy as np
import random
import os

class ReplayMemory:

    def __init__(self, config):
        self.config = config
        self.actions = np.empty((self.config.mem_size), dtype=np.int32)
        self.rewards = np.empty((self.config.mem_size), dtype=np.int32)
        # Screens are dtype=np.uint8 which saves massive amounts of memory, however the network expects state inputs
        # to be dtype=np.float32. Remember this every time you feed something into the network
        self.screens = np.empty((self.config.mem_size, self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.terminals = np.empty((self.config.mem_size,), dtype=np.float16)
        self.count = 0
        self.current = 0
        self.dir_save = config.dir_save + "memory/"

        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)

    def save(self):
        np.save(self.dir_save + "screens.npy", self.screens)
        np.save(self.dir_save + "actions.npy", self.actions)
        np.save(self.dir_save + "rewards.npy", self.rewards)
        np.save(self.dir_save + "terminals.npy", self.terminals)

    def load(self):
        self.screens = np.load(self.dir_save + "screens.npy")
        self.actions = np.load(self.dir_save + "actions.npy")
        self.rewards = np.load(self.dir_save + "rewards.npy")
        self.terminals = np.load(self.dir_save + "terminals.npy")



class DQNReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DQNReplayMemory, self).__init__(config)

        self.pre = np.empty((self.config.batch_size, self.config.history_len, self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.post = np.empty((self.config.batch_size, self.config.history_len, self.config.screen_height, self.config.screen_width), dtype=np.uint8)

    def getState(self, index):

        index = index % self.count
        if index >= self.config.history_len - 1:
            a = self.screens[(index - (self.config.history_len - 1)):(index + 1), ...]
            return a
        else:
            indices = [(index - i) % self.count for i in reversed(range(self.config.history_len))]
            return self.screens[indices, ...]

    def add(self, screen, reward, action, terminal):
        assert screen.shape == (self.config.screen_height, self.config.screen_width)

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current] = screen
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size

    def sample_batch(self):
        assert self.count > self.config.history_len

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.history_len, self.count-1)

                if index >= self.current and index - self.config.history_len < self.current:
                    continue

                if self.terminals[(index - self.config.history_len): index].any():
                    continue
                break
            self.pre[len(indices)] = self.getState(index - 1)
            self.post[len(indices)] = self.getState(index)
            indices.append(index)

        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return self.pre, actions, rewards, self.post, terminals

class DRQNReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DRQNReplayMemory, self).__init__(config)

        self.timesteps = np.empty((self.config.mem_size), dtype=np.int32)
        self.states = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1, self.config.screen_height, self.config.screen_width), dtype=np.uint8)

    def add(self, screen, reward, action, terminal, t):
        assert screen.shape == (self.config.screen_height, self.config.screen_width)

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current] = screen
        self.timesteps[self.current] = t
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size


    def getState(self, index):
        a = self.screens[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a


    def sample_batch(self):
        assert self.count > self.config.min_history + self.config.states_to_update

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.min_history, self.count-1)
                if index >= self.current and index - self.config.min_history < self.current:
                    continue
                if index < self.config.min_history + self.config.states_to_update + 1:
                    continue
                if self.timesteps[index] < self.config.min_history + self.config.states_to_update:
                    continue
                break
            self.states[len(indices)] = self.getState(index)
            indices.append(index)


        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return self.states, actions, rewards, terminals
