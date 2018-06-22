import retro
import numpy as np
from networks.dqn import DQN
from tqdm import tqdm
from PIL import Image
from history import History
from replay_memory import ReplayMemory
from skimage.transform import resize
import matplotlib.pyplot as plt
class Agent():


    def __init__(self, batch_size=128,history_len=4,mem_size=20000, network_type="dqn", frame_skip=4, epsilon_start=1, epsilon_end=0.1, epsilon_decay_episodes=10000, screen_height=80, screen_width=112, train_freq=2, update_freq=10):
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.frame_skip = frame_skip
        self.history_len = history_len
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.epsilon_decay = (epsilon_start-epsilon_end)/epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.train_freq = train_freq
        self.update_freq = update_freq

        self.env = retro.make(game='Airstriker-Genesis', state='Level1.state',record='.')
        self.net = DQN(2**self.env.action_space.n,self.history_len, 80, 112)
        self.history = History(self.batch_size, self.history_len, self.screen_height, self.screen_width)
        self.replay_memory = ReplayMemory(self.mem_size, self.screen_height, self.screen_width, self.batch_size, self.history_len)
        self.net.build()

    def policy(self, state):
        if np.random.rand()< self.epsilon:
            return self.env.action_space.sample()
        else:
            q, a = self.net.predict_batch([state])
            return self._to_actionspace(a[0])

    def observe(self, screen, action, reward, done):
        screen = self._preprocess(screen)
        action = self._to_action_int(action)
        self.history.add(screen)
        self.replay_memory.add(screen, reward, action, done)


    def run_episode(self):
        ob = self.env.reset()
        if self.i == 0:
            for _ in range(self.history_len):
                self.history.add(self._preprocess(ob))
        t = 0
        done = False
        skip = 0
        ac =  self.policy(self.history.get())
        while not done:
            ob_, reward, done, info = self.env.step(ac)
            if skip == self.frame_skip: # TODO this might be buggy
                self.observe(ob_, ac, reward, done) #komisch, hier den nächsten screen zu nehmen...
                t += 1
                skip = 0
                ac =  self.policy(self.history.get())
            else:
                skip += 1
            # if t % 3 == 0:
            #     self.env.render()
            ob = ob_

    def train(self):
        for self.i in tqdm(range(200000)):
            self.run_episode()
            if self.i < self.epsilon_decay_episodes:
                self.epsilon -= self.epsilon_decay
            if self.i % self.train_freq == 0:
                for i in range(10):
                    self.net.train_on_batch(*self.replay_memory.sample_batch())
            if self.i % self.update_freq == 0:
                self.net.update_target()


    def _preprocess(self, ob):
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        return rgb2gray(resize(ob, (80, 112, 3)))

    def _to_actionspace(self, action):
        return list(map(int, list("{0:012b}".format(action))))

    def _to_action_int(self, action):
        return int("".join(map(str, action)), 2)
