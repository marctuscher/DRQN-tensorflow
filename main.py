from src.dqn_agent import DQNAgent
from src.config import Config

class Main():

    def __init__(self):
        self.agent = DQNAgent(Config())
    def train(self, steps):
        self.agent.train(steps)

    def play(self, episodes):
        self.agent.play(episodes)

main = Main()
main.train(10000000)
