from scipy.misc import imresize as resize
import numpy as np
def rgb2gray(screen):
    return np.mean(screen, axis=2)
