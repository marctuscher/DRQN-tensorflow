import gym
from utils import resize, rgb2gray
import numpy as np

class GymWrapper():

    def __init__(self, env_name, screen_width, screen_height):
        self.env = gym.make(env_name)
        self.screen_width, self.screen_height = screen_width, screen_height
        self._screen = None
        self.reward = 0
        self.terminal = True
        self.info = {'ale.lives': 0}
        self.random_start = 30
        self.action_repeat = 1

    def new_game(self):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game()
        for _ in range(np.random.randint(0, self.random_start)):
            self._step(0)
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, self.info = self.env.step(action)

    def random_step(self):
        return self.env.action_space.sample()

    def act(self, action):
        cumulated = 0
        start_lives = self.lives
        for _ in range(self.action_repeat):
            self._step(action)
            cumulated = cumulated +self.reward
            if start_lives > self.lives:
                self.terminal = True
            if self.terminal:
                break
        self.reward = cumulated

    @property
    def screen(self):
        return resize(rgb2gray(self._screen), (self.screen_height, self.screen_width))

    @property
    def lives(self):
        return self.info['ale.lives']


