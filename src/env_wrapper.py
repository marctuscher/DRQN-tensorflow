import gym
import retro
from src.utils import resize, rgb2gray
import numpy as np



class GymWrapper():

    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.screen_width, self.screen_height = config.screen_width, config.screen_height
        self.reward = 0
        self.terminal = True
        self.info = {'ale.lives': 0}
        self.env.env.frameskip = config.frame_skip
        self.random_start = config.random_start
        self.action_space = self.env.action_space
        self._screen = np.empty((210, 160), dtype=np.uint8)

    def new_game(self):
        if self.lives == 0:
            self.env.reset()
        self._step(0)
        self.reward = 0
        self.action = 0

    def new_random_game(self):
        self.new_game()
        for _ in range(np.random.randint(0, self.random_start)):
            self._step(0)


    def _step(self, action):
        self.action = action
        _, self.reward, self.terminal, self.info = self.env.step(action)


    def random_step(self):
        return self.action_space.sample()

    def act(self, action):
        lives_before = self.lives
        self._step(action)
        if self.lives < lives_before:
            self.terminal = True


    def act_play(self, action):
        lives_before = self.lives
        self._step(action)
        self.env.render()
        if self.lives < lives_before:
            self.terminal = True

    @property
    def screen(self):
        self._screen = self.env.env.ale.getScreenGrayscale(self._screen)
        a = resize(self._screen ,(self.screen_height, self.screen_width))
        return a

    @property
    def lives(self):
        return self.info['ale.lives']



class RetroWrapper():

    def __init__(self, config):
        self.env = retro.make(game=config.env_name, state=config.state, use_restricted_actions=2)
        self.screen_width, self.screen_height = config.screen_width, config.screen_height
        self.reward = 0
        self.terminal = True
        self.info = {'lives': 0}
        #self.env.env.frameskip = config.frame_skip
        self.random_start = config.random_start
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))
        print(self.action_space.sample())

    def new_game(self):
        if self.lives == 0:
            self.env.reset()
        self._step(0)
        self.reward = 0
        self.action = 0

    def new_random_game(self):
        self.new_game()
        for _ in range(np.random.randint(0, self.random_start)):
            self._step(0)


    def _step(self, action):
        self.action = action
        self._screen, self.reward, self.terminal, self.info = self.env.step(action)

    def random_step(self):
        return self.action_space.sample()

    def act(self, action):
        lives_before = self.lives
        self._step(action)
        if self.lives < lives_before:
            self.terminal = True


    def act_play(self, action):
        lives_before = self.lives
        self._step(action)
        self.env.render()
        if self.lives < lives_before:
            self.terminal = True

    @property
    def screen(self):
        a = resize(rgb2gray(self._screen) ,(self.screen_height, self.screen_width))
        return a

    @property
    def lives(self):
        return self.info['lives']
