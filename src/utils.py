from scipy.misc import imresize as resize
import numpy as np
def rgb2gray(screen):
    return np.dot(screen[..., :3], [0.299,0.587,0.114])/255
    #return np.mean(screen, axis=2)