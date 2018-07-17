from src.dqn_agent import DQNAgent
from src.drqn_agent import DRQNAgent
from src.config import RetroConfig, GymConfig
import sys
class Main():

    def __init__(self, net_type, conf):
        if net_type == "drqn":
            self.agent = DRQNAgent(conf)
        else:
            self.agent = DQNAgent(conf)
    def train(self, steps):
        self.agent.train(steps)

    def play(self, episodes):
        self.agent.play(episodes)

if sys.argv[3] == "sonic":
    conf = RetroConfig
else:
    conf = GymConfig
main = Main(sys.argv[1], conf)


if sys.argv[2] == "play":
    net_path = sys.argv[3]
    main.play(net_path)
else:
    main.train(10000000)
