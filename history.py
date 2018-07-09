import numpy as np
import os

class History:

    def __init__(self, batch_size, history_len, screen_height, screen_width):
        self.batch_size = batch_size
        self.history_len = history_len
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.history = np.zeros((self.history_len, self.screen_height, self.screen_width), dtype=np.uint8)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history
