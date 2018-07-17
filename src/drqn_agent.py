from src.agent import BaseAgent
from src.replay_memory import DRQNReplayMemory
from src.networks.drqn import DRQN
import numpy as np
from tqdm import tqdm

class DRQNAgent(BaseAgent):

    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)
        self.replay_memory = DRQNReplayMemory(config)
        self.net = DRQN(self.env_wrapper.action_space.n, config)
        self.net.build()
        self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

    def observe(self, t):
        reward = max(self.min_reward, min(self.max_reward, self.env_wrapper.reward))
        self.screen = self.env_wrapper.screen
        self.replay_memory.add(self.screen, reward, self.env_wrapper.action, self.env_wrapper.terminal, t)
        if self.i < self.config.epsilon_decay_episodes:
            self.epsilon -= self.config.epsilon_decay
        if self.i % self.config.train_freq == 0 and self.i > self.config.train_start:
            states, action, reward, terminal = self.replay_memory.sample_batch()
            q, loss= self.net.train_on_batch_target(states, action, reward, terminal, self.i)
            self.total_q += q
            self.total_loss += loss
            self.update_count += 1
        if self.i % self.config.update_freq == 0:
            self.net.update_target()

    def policy(self, state):
        self.random = False
        if np.random.rand() < self.epsilon:
            self.random = True
            return self.env_wrapper.random_step()
        else:
            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run([self.net.q_action, self.net.state_output_c, self.net.state_output_h],{
                self.net.state : [[state]],
                self.net.c_state_train: self.lstm_state_c,
                self.net.h_state_train: self.lstm_state_h
            })
            return a[0]


    def train(self, steps):
        render = False
        self.env_wrapper.new_random_game()
        num_game, self.update_count, ep_reward = 0,0,0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        ep_rewards, actions = [], []
        t = 0
        self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single

        for self.i in tqdm(range(self.i, steps)):
            state = self.env_wrapper.screen/255.0
            action = self.policy(state)
            self.env_wrapper.act(action)
            if self.random:
                self.lstm_state_c, self.lstm_state_h = self.net.sess.run([self.net.state_output_c, self.net.state_output_h], {
                    self.net.state: [[state]],
                    self.net.c_state_train : self.lstm_state_c,
                    self.net.h_state_train: self.lstm_state_h
                })
            self.observe(t)
            if self.env_wrapper.terminal:
                t = 0
                self.env_wrapper.new_random_game()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
                self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
            else:
                ep_reward += self.env_wrapper.reward
                t += 1
            actions.append(action)
            total_reward += self.env_wrapper.reward

            if self.i >= self.config.train_start:
                if self.i % self.config.test_step == self.config.test_step -1:
                    avg_reward = total_reward / self.config.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    sum_dict = {
                        'average_reward': avg_reward,
                        'average_loss': avg_loss,
                        'average_q': avg_q,
                        'ep_max_reward': max_ep_reward,
                        'ep_min_reward': min_ep_reward,
                        'ep_num_game': num_game,
                        'learning_rate': self.net.learning_rate,
                        'ep_rewards': ep_rewards,
                        'ep_actions': actions
                    }
                    self.net.inject_summary(sum_dict, self.i)
                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

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

