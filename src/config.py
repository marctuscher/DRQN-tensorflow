

class Config(object):

    #env_name = "SonicAndKnuckles3-Genesis"
    env_name = "Breakout-v0"
    train_steps = 10000000
    batch_size = 32
    history_len = 4
    mem_size = 800000
    frame_skip = 4
    epsilon_start = 1.0
    epsilon_end = 0.1
    max_steps = 10000
    epsilon_decay_episodes = 1000000
    screen_height = 84
    screen_width = 84
    train_freq = 4
    update_freq = 10000
    train_start = 20000
    dir_save = "saved_session/"
    restore = False
    epsilon_decay = float((epsilon_start - epsilon_end))/float(epsilon_decay_episodes)
    random_start = 10
    test_step = 50000


    gamma = 0.99
    learning_rate_minimum = 0.00025
    lr_method = "rmsprop"
    learning_rate = 0.00025
    lr_decay = 0.97
    keep_prob = 0.8

    num_lstm_layers = 1
    lstm_size = 512
    min_history = 4
    states_to_update = 4


    #state="HydrocityZone.Act1"
    state = None