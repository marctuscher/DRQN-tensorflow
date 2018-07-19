from src.dqn_agent import DQNAgent
from src.drqn_agent import DRQNAgent
from src.config import RetroConfig, GymConfig
import sys

import argparse

class Main():

    def __init__(self, net_type, conf):
        if net_type == "drqn":
            self.agent = DRQNAgent(conf)
        else:
            self.agent = DQNAgent(conf)

    def train(self, steps):
        self.agent.train(steps)

    def play(self, episodes, net_path):
        self.agent.play(episodes, net_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRQN")
    parser.add_argument("--gym", type=str, default="gym", help="Type of the environment. Can either be 'gym' or 'retro'")
    parser.add_argument("--network_type", type=str, default="dqn", help="Type of the network to build, can either be 'dqn' or 'drqn'")
    parser.add_argument("--env_name", type=str, default="Breakout-v0", help="Name of the gym/retro environment used to train the agent")
    parser.add_argument("--retro_state", type=str, default="Start", help="Name of the state (level) to start training. This is only necessary for retro envs")
    parser.add_argument("--train", type=str, default="True", help="Whether to train a network or to play with a given network")
    parser.add_argument("--model_dir", type=str, default="saved_session/net/", help="directory to save the model and replay memory during training")
    parser.add_argument("--net_path", type=str, default="", help="path to checkpoint of model")
    parser.add_argument("--steps", type=int, default=50000000, help="number of frames to train")
    args, remaining = parser.parse_known_args()

    if args.gym == "gym":
        conf = GymConfig()
        conf.env_name = args.env_name
    else:
        conf = RetroConfig()
        conf.env_name = args.env_name
        conf.state = args.retro_state
    conf.network_type = args.network_type
    conf.train = args.train
    conf.dir_save = args.model_dir
    conf.train_steps = args.steps
    main = Main(conf.network_type, conf)

    if conf.train == "True":
        print(conf.train)
        main.train(conf.train_steps)
    else:
        assert args.net_path != "", "Please specify a net_path using the option --net_path"
        main.play(100000, args.net_path)



