from src.agent import Agent
from src.config import Config

class Main():

    def __init__(self):
        self.agent = Agent(Config())
    def train(self, steps):
        self.agent.train(steps)

    def play(self, episodes):
        self.agent.play(episodes)

main = Main()
main.train(10000000)
