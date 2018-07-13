from src.env_wrapper import GymWrapper
import numpy as np
from src.networks.dqn import DQN
from tqdm import tqdm
from src.history import History
from src.replay_memory import ReplayMemory

class Agent():

    def __init__(self, config):
        self.config = config
        self.env_wrapper = GymWrapper(config)
        print("actionspace_n: {}".format(self.env_wrapper.env.action_space.n))
        self.net = DQN(self.env_wrapper.env.action_space.n, config)
        self.net.build()
        self.history = History(config)
        self.replay_memory = ReplayMemory(config)
        self.net.add_summary(['total_reward', 'episode_num', 'avg_q', 'episode_len', 'epsilon', 'learning_rate'])
        self.rewards = 0
        self.lens = 0
        self.epsilon = config.epsilon_start
        if self.config.restore:
            self.load()
        else:
            self.i = 0

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
        screen = self.env_wrapper.screen
        self.history.add(screen)
        self.replay_memory.add(screen, self.env_wrapper.reward, action, self.env_wrapper.terminal)

    def train(self, steps):
        self.env_wrapper.new_game()
        episode_len, reward, counter = 0.0, 0.0, 1.0
        episode_num = 0.0
        total_q = 0.0
        train_count = 0.0
        render = False
        for _ in range(self.config.history_len):
            self.history.add(self.env_wrapper.screen)
        for self.i in tqdm(range(self.i, steps)):
            episode_steps = 0
            action = self.policy(self.history.get())
            self.env_wrapper.act(action)
            self.observe(action)
            if episode_steps > self.config.max_steps:
                self.env_wrapper.terminal = True
                self.env_wrapper.reward = -200
            if self.env_wrapper.terminal:
                episode_steps = 0
                self.lens += 1
                episode_num += 1
                self.rewards += self.env_wrapper.reward
                counter += 1
                self.env_wrapper.new_game()
            else:
                episode_steps += 1
                self.lens += 1
                self.rewards += self.env_wrapper.reward
                #avq_q += q
            if self.i < self.config.epsilon_decay_episodes and self.i > self.config.train_start:
                self.epsilon -= self.config.epsilon_decay
            if self.i % self.config.train_freq == 0 and self.i > self.config.train_start:
                state, action, reward, state_, terminal = self.replay_memory.sample_batch()
                total_q += self.net.train_on_batch_target(state, action, reward, state_, terminal, self.i)
                train_count += 1.0
            if self.i % self.config.update_freq == 0:
                self.net.update_target()
            if self.i % 1000 == 0 and self.i > self.config.train_start:
                sum_dict = {
                            'total_reward': float(self.rewards/counter),
                            'episode_len': float(self.lens/counter),
                    'episode_num': float(episode_num),
                            'epsilon': self.epsilon,
                            'learning_rate': self.net.learning_rate,
                            'avg_q': total_q / train_count
                            }
                counter = 0
                train_count = 0.0
                total_q = 0.0
                self.lens, self.rewards = 0.0, 0.0
                episode_num = 0.0
                self.net.inject_summary(sum_dict, self.i)
            if self.i % 500000 == 0 and self.i > 0:
                j = 0
                self.save()
            if self.i % 100000 == 0:
                j = 0
                render = True

            if render:
                self.env_wrapper.env.render()
                j += 1
                if j == 1000:
                    render = False
    def save(self):
        self.replay_memory.save()
        self.net.save_session()
        np.save(self.config.dir_save+'step.npy', self.i)

    def load(self):
        self.replay_memory.load()
        self.net.restore_session()
        self.i = np.load(self.config.dir_save+'step.npy')

    def play(self, episodes):
        self.env_wrapper.new_game()
        i = 0
        for _ in range(self.config.history_len):
            self.history.add(self.env_wrapper.screen)
        episode_steps = 0
        while i < episodes:
            a = self.net.q_action.eval({
                self.net.state : [self.history.get()],
                self.net.dropout: 1.0
            }, session=self.net.sess)
            self.env_wrapper.act_play(a[0])
            self.history.add(self.env_wrapper.screen)
            episode_steps += 1
            if episode_steps > self.config.max_steps:
                self.env_wrapper.terminal = True
            if self.env_wrapper.terminal:
                episode_steps = 0
                i += 1
                self.env_wrapper.new_game()
                for _ in range(self.config.history_len):
                    self.history.add(self.env_wrapper.screen)
