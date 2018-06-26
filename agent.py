import retro
import numpy as np
from networks.dqn import DQN
from tqdm import tqdm
from PIL import Image
from history import History
from replay_memory import ReplayMemory
from cv2 import resize
from skimage import color
import matplotlib.pyplot as plt
from numba import jit
class Agent():


    def __init__(self, batch_size=32,history_len=4,mem_size=40000, network_type="dqn", frame_skip=4, epsilon_start=1, epsilon_end=0.2, epsilon_decay_episodes=50000, screen_height=82, screen_width=82, train_freq=1, update_freq=4):
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
        self.pred_before_train = True
        self.env = retro.make(game='Breakout-Atari2600', state='Start',record='.')
        print("actionspace_n: {}".format(self.env.action_space.n))
        self.net = DQN(2**self.env.action_space.n,self.history_len, self.screen_width, self.screen_height, pred_before_train=self.pred_before_train)
        self.history = History(self.batch_size, self.history_len, self.screen_height, self.screen_width)
        self.replay_memory = ReplayMemory(self.mem_size, self.screen_height, self.screen_width, self.batch_size, self.history_len)
        self.net.build()
        self.net.add_summary(['total_reward', 'avg_reward', 'avg_q', 'episode_len', 'epsilon', 'learning_rate'])

    def policy(self, state):
        if np.random.rand()< self.epsilon:
            return 0 ,self.env.action_space.sample()
        else:
            q, a = self.net.predict_batch([state])
            return np.max(q[0]), self._to_actionspace(a[0])

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
        t, episode_len, total_reward, total_q, skip, done =0, 0.0, 0.0, 0.0, 0.0, False
        q, ac =  self.policy(self.history.get())
        while not done:
            rewards = 0
            for _ in range(self.frame_skip):
                ob_, reward, done, info = self.env.step(ac)
                rewards += reward
            self.observe(ob_, ac, rewards, done)
            q, ac =  self.policy(self.history.get())
            episode_len += 1
            total_reward += rewards
            total_q += q
        return episode_len, total_reward, total_q

    def train(self):
        for self.i in tqdm(range(200000)):
            episode_len, total_reward, total_q = self.run_episode()
            if self.i % self.train_freq == 0:
                for i in range(1):
                    # if self.pred_before_train:
                    self.net.train_on_batch_target(*self.replay_memory.sample_batch())
                    # else:
                        # self.net.train_on_batch(*self.replay_memory.sample_batch())
                    if self.net.train_steps < self.epsilon_decay_episodes:
                        self.epsilon -= self.epsilon_decay
                sum_dict = {'total_reward': float(total_reward),
                            'avg_reward': float(total_reward/episode_len),
                            'avg_q': float(total_q/episode_len),
                            'episode_len': float( episode_len),
                            'epsilon': self.epsilon,
                            'learning_rate': self.net.learning_rate
                }
                self.net.inject_summary(sum_dict)
            if self.i % self.update_freq == 0:
                self.net.update_target()
            if self.i % 500 == 0:
                self.net.save_session()

    def _preprocess(self, ob):
        def rgb2gray(ob):
            return np.dot(ob[...,:3], [0.299, 0.587, 0.114])
        return resize(rgb2gray(ob)/255, (self.screen_height, self.screen_width))

    def _to_actionspace(self, action):
        return list(map(int, list(format(action, '0{}b'.format(self.env.action_space.n)))))

    def _to_action_int(self, action):
        return int("".join(map(str, action)), 2)
