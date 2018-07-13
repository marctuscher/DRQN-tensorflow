from scipy.misc import imresize as resize
import json
import numpy as np
def rgb2gray(screen):
    return np.dot(screen[..., :3], [0.299,0.587,0.114])/255
    #return np.mean(screen, axis=2)


def load_config(config_file):
    pass

def save_config(config_file, config_dict):
    with open(config_file, 'w') as fp:
        json.dump(config_dict, fp)



