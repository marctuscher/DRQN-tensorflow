import numpy as np
import random
from numba import jit

class ReplayMemory:

    def __init__(self, mem_size, screen_height, screen_width, batch_size, history_len):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.history_len = history_len

        self.actions = np.empty((self.mem_size), dtype=np.uint16)
        self.rewards = np.empty((self.mem_size), dtype=np.uint32)
        self.screens = np.empty((self.mem_size, self.screen_height, self.screen_width), dtype=np.uint8)
        self.terminals = np.empty((self.mem_size,), dtype=np.bool)
        self.count = 0
        self.current = 0

        self.pre = np.empty((self.batch_size, self.history_len, self.screen_height, self.screen_width), dtype=np.float16)
        self.post = np.empty((self.batch_size, self.history_len, self.screen_height, self.screen_width), dtype=np.float16)

    def add(self, screen, reward, action, terminal):
        assert screen.shape == (self.screen_height, self.screen_width)

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.mem_size

    def getState(self, index):

        index = index % self.count
        if index >= self.history_len - 1:
            return self.screens[(index - (self.history_len -1)): (index + 1)]
        else:
            indices = [(index - i) % self.count for i in reversed(range(self.history_len))]
            return self.screens[indices]


    def sample_batch(self):
        assert self.count > self.history_len

        indices = []
        while len(indices) < self.batch_size:

            while True:
                index = random.randint(self.history_len, self.count-1)

                if index >= self.current and index - self.history_len < self.current:
                    continue

                if self.terminals[(index - self.history_len): index].any():
                    continue
                break
            self.pre[len(indices)] = self.getState(index - 1)
            self.post[len(indices)] = self.getState(index)
            indices.append(index)

        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return self.pre, actions, rewards, self.post, terminals
