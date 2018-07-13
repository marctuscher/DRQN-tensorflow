

class Config(object):

    env_name = "Breakout-v0"
    train_steps = 10000000
    batch_size = 64
    history_len = 4
    mem_size = 500000
    frame_skip = 4
    epsilon_start = 1
    epsilon_end = 0.1
    max_steps = 10000
    epsilon_decay_episodes = 500000
    screen_height = 84
    screen_width = 84
    train_freq = 4
    update_freq = 10000
    train_start = 50000
    dir_save = "saved_session/"
    restore = False
    epsilon_decay = (epsilon_start - epsilon_end)/epsilon_decay_episodes


    gamma = 0.99
    learning_rate_minimum = 0.00025
    lr_method = "rmsprop"
    learning_rate = 0.003
    lr_decay = 0.97
    keep_prob = 0.8
