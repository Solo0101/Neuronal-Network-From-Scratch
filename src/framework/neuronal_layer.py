#import numpy as np
import cupy as cp

class Layer:
    def __init__(self, inputs, number_of_neurons):
        self.inputs = inputs
        self.neurons = number_of_neurons
        self.weights = cp.random.rand(number_of_neurons, len(inputs)) - 0.5
        self.biases = cp.random.rand(number_of_neurons, ) - 0.5

    def forward(self):
        return self.weights.dot(self.inputs) + self.biases
