import retro
import numpy as np
from networks.dqn import DQN
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
class Agent():


    def __init__(self, mem_size=50000, network_type="dqn", frame_skip=4, epsilon_start=1, epsilon_end=0.1, epsilon_decay_episodes=10000):
        self.env = retro.make(game='Airstriker-Genesis', state='Level1.state',record='.')
        self.memory_states = np.zeros((mem_size,1, 80, 112))
        self.memory_states_ = np.zeros((mem_size,1, 80, 112))
        self.memory_actions = np.zeros((mem_size,))
        self.memory_rewards = np.zeros((mem_size,))
        self.memory_t = np.zeros((mem_size,))
        self.last_insert = -1
        self.net = DQN(2**self.env.action_space.n, 80, 112)
        self.net.build()
        self.mem_size = mem_size
        self.frame_skip = frame_skip
        self.epsilon_decay = (epsilon_start-epsilon_end)/epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self._filled_once = False

    def policy(self, state):
        if np.random.rand()< self.epsilon:
            return self.env.action_space.sample()
        else:
            q, a = self.net.predict_batch([[self._preprocess(state)]])
            return self._to_actionspace(a[0])


    def safe(self, ob, ac, reward, ob_, t):
        self.last_insert = (self.last_insert + 1) % self.mem_size
        if self.last_insert == self.mem_size -1:
            self._filled_once = True
        self.memory_states[self.last_insert][0] = self._preprocess(ob)
        self.memory_states_[self.last_insert][0] = self._preprocess(ob_)
        self.memory_actions[self.last_insert] = self._to_action_int(ac)
        self.memory_rewards[self.last_insert] = reward
        self.memory_t[self.last_insert] = t

    def run_episode(self):
        ob = self.env.reset()
        t = 0
        done = False
        skip = 0
        while not done:
            ac =  self.policy(ob)
            ob_, reward, done, info = self.env.step(ac)
            if skip == self.frame_skip or done: # TODO this might be buggy 
                self.safe(ob, ac, reward, ob_, t)
                t += 1
                skip = 0 
            else:
                skip += 1
            ob = ob_

    def train(self):
        for i in tqdm(range(2000)):
            self.run_episode()
            if i < self.epsilon_decay_episodes:
                self.epsilon -= self.epsilon_decay
            if i != 0 and i % 85 == 0 and self._filled_once:
                print("last insert: ", self.last_insert)
                data=list(zip(*[self.memory_states, self.memory_actions, self.memory_rewards, self.memory_states_]))
                self.net.train(data)


    def _preprocess(self, ob):
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        return rgb2gray(resize(ob, (80, 112, 3)))

    def _to_actionspace(self, action):
        return list(map(int, list("{0:012b}".format(action))))

    def _to_action_int(self, action):
        return int("".join(map(str, action)), 2)
