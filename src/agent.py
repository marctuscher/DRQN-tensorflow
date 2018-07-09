from src.env_wrapper import GymWrapper
import numpy as np
from src.networks.dqn import DQN
from tqdm import tqdm
from src.history import History
from src.replay_memory import ReplayMemory

class Agent():

    def __init__(self, train_steps=10000000,batch_size=64, history_len=4, mem_size=700000, frame_skip=4, epsilon_start=1, epsilon_end=0.1,
                 epsilon_decay_episodes=100000, screen_height=84, screen_width=84, train_freq=4, update_freq=2000,
                 learn_start=10000, dir_save="saved_session/", restore=False, train_start=50000):
        self.learn_start = learn_start
        self.train_steps = train_steps
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
        self.dir_save = dir_save
        self.train_start = train_start
        self.env_wrapper = GymWrapper('BreakoutDeterministic-v0', self.screen_height, self.screen_width, frame_skip=frame_skip)
        print("actionspace_n: {}".format(self.env_wrapper.env.action_space.n))
        self.net = DQN(self.env_wrapper.env.action_space.n, self.history_len, self.screen_width, self.screen_height,
                       pred_before_train=self.pred_before_train, dir_save= dir_save)
        self.history = History(self.batch_size, self.history_len, self.screen_height, self.screen_width)
        self.replay_memory = ReplayMemory(self.mem_size, self.screen_height, self.screen_width, self.batch_size,
                                          self.history_len, dir_save= dir_save)
        self.net.build()
        self.i = 0
        self.net.add_summary(['total_reward', 'episode_num', 'avg_q', 'episode_len', 'epsilon', 'learning_rate'])
        self.rewards = 0
        self.lens = 0
        if restore:
            self.load()


    def policy(self, state):
        if np.random.rand() < self.epsilon:
            return self.env_wrapper.random_step()
        else:
            a = self.net.q_action.eval({
                self.net.state : [state],
                self.net.dropout: 1.0
                }, session=self.net.sess)
            return a[0]

    def observe(self, action):
        self.history.add(self.env_wrapper.screen())
        self.replay_memory.add(self.env_wrapper.screen(), self.env_wrapper.reward, action, self.env_wrapper.terminal)

    def train(self, steps):
        self.env_wrapper.new_random_game()
        episode_len, reward, counter = 0.0, 0.0, 1.0
        episode_num = 0.0

        for _ in range(self.history_len):
            self.history.add(self.env_wrapper.screen())
        for self.i in tqdm(range(self.i, steps)):
            action = self.policy(self.history.get())
            self.env_wrapper.act(action)
            self.observe(action)
            if self.env_wrapper.terminal:
                self.lens += 1
                episode_num += 1
                self.rewards += self.env_wrapper.reward
                counter += 1
                self.env_wrapper.new_random_game()
            else:
                self.lens += 1
                self.rewards += self.env_wrapper.reward
                #avq_q += q
            if self.i < self.epsilon_decay_episodes:
                self.epsilon -= self.epsilon_decay
            if self.i % self.train_freq == 0 and self.i > self.train_start:
                self.net.train_on_batch_target(*self.replay_memory.sample_batch())
            if self.i % self.update_freq == 0:
                self.net.update_target()
            if self.i % 1000 == 0 and self.i > self.train_start:
                sum_dict = {
                            'total_reward': float(self.rewards/counter),
                            'episode_len': float(self.lens/counter),
                    'episode_num': float(episode_num/counter),
                            'epsilon': self.epsilon,
                            'learning_rate': self.net.learning_rate
                            }
                counter = 0
                self.lens, self.rewards = 0.0, 0.0
                episode_num = 0.0
                self.net.inject_summary(sum_dict)
            if self.i % 500000 == 0:
                self.save()

    def save(self):
        self.replay_memory.save()
        self.net.save_session()
        np.save(self.dir_save+'step.npy', self.i)

    def load(self):
        self.replay_memory.load()
        self.net.restore_session()
        self.i = np.load(self.dir_save+'step.npy')
