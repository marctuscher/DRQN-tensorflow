from src.dqn_agent import DQNAgent
from src.drqn_agent import DRQNAgent
from src.config import Config
import sys
class Main():

    def __init__(self, net_type):
        if net_type == "drqn":
            self.agent = DRQNAgent(Config())
        else:
            self.agent = DQNAgent(Config())
    def train(self, steps):
        self.agent.train(steps)

    def play(self, episodes):
        self.agent.play(episodes)

main = Main(sys.argv[1])
main.train(10000000)
