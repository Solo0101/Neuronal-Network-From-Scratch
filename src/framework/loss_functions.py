#import numpy as np
import cupy as cp
from src.enviroment.constants import output_layer_neurons_number


def loss(output, y):
    y_onehot = cp.zeros((y.size, output_layer_neurons_number))
    y_onehot[cp.arange(y.size), y] = 1
    #y_onehot = y_onehot.T
    #return np.square(output - y_onehot)
    return (output - y_onehot).T