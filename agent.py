import retro
from env_wrapper import GymWrapper
import numpy as np
from networks.dqn import DQN
from tqdm import tqdm
from PIL import Image
from history import History
from replay_memory import ReplayMemory

class Agent():

    def __init__(self, batch_size=32, history_len=4, mem_size=80000, frame_skip=4, epsilon_start=1, epsilon_end=0.2,
                 epsilon_decay_episodes=1000000, screen_height=42, screen_width=42, train_freq=4, update_freq=10000,
                 learn_start=10000):
        self.learn_start = learn_start
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.frame_skip = frame_skip
        self.history_len = history_len
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.train_freq = train_freq
        self.update_freq = update_freq
        self.pred_before_train = True
        self.env_wrapper = GymWrapper('Breakout-v0', self.screen_height, self.screen_width)
        print("actionspace_n: {}".format(self.env_wrapper.env.action_space.n))
        self.net = DQN(self.env_wrapper.env.action_space.n, self.history_len, self.screen_width, self.screen_height,
                       pred_before_train=self.pred_before_train)
        self.history = History(self.batch_size, self.history_len, self.screen_height, self.screen_width)
        self.replay_memory = ReplayMemory(self.mem_size, self.screen_height, self.screen_width, self.batch_size,
                                          self.history_len)
        self.net.build()
        self.i = 0
        self.net.add_summary(['total_reward', 'avg_reward', 'avg_q', 'episode_len', 'epsilon', 'learning_rate'])

    def policy(self, state):
        if np.random.rand() < self.epsilon:
            return 0, self.env_wrapper.random_step()
        else:
            q, a = self.net.predict_batch([state])
            return np.max(q[0]), a[0]

    def observe(self, action):
        self.history.add(self.env_wrapper.screen)
        self.replay_memory.add(self.env_wrapper.screen, self.env_wrapper.reward, action, self.env_wrapper.terminal)

    def train(self):
        self.env_wrapper.new_random_game()
        total_reward, avg_reward, episode_len, avq_q, eps = 0, 0, 0, 0, 0
        if self.i == 0:
            for _ in range(self.history_len):
                self.history.add(self.env_wrapper.screen)
        for self.i in tqdm(range(10000000)):
            q, action = self.policy(self.history.get())
            self.env_wrapper.act(action)
            self.observe(action)
            if self.env_wrapper.terminal:
                episode_len += 1
                avq_q /= episode_len
                avg_reward = total_reward / episode_len
                sum_dict = {'total_reward': float(total_reward),
                            'avg_reward': float(avg_reward),
                            'avg_q': float(avq_q),
                            'episode_len': float(episode_len),
                            'epsilon': self.epsilon,
                            'learning_rate': self.net.learning_rate
                            }
                self.net.inject_summary(sum_dict)
                episode_len = 0
                total_reward = 0
                avq_q = 0
                self.env_wrapper.new_random_game()
            else:
                episode_len += 1
                total_reward += self.env_wrapper.reward
                avq_q += q
            if self.i < self.epsilon_decay_episodes:
                self.epsilon -= self.epsilon_decay
            if self.i % self.train_freq == 0 and self.i > self.learn_start:
                self.net.train_on_batch_target(*self.replay_memory.sample_batch())
            if self.i % self.update_freq == 0:
                self.net.update_target()
            if self.i % 500000 == 0:
                self.net.save_session()
