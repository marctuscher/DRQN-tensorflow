from src.agent import Agent

class Main():

    def __init__(self):
        self.agent = Agent()
    def train(self, steps):
        self.agent.train(steps)

    def play(self, episodes):
        self.agent.play(episodes)

main = Main()
main.train(10000000)
