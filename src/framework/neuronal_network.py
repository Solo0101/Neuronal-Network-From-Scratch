from src.framework.neuronal_layer import Layer
from src.framework.activation_functions import *
from src.enviroment.constants import *
from src.framework.loss_functions import *


class Network:
    def __init__(self, number_of_layers, layers_layout: cp.array):
        self.number_of_layers = number_of_layers
        self.layers_layout = layers_layout
        self.layers = []
        self.outputs = cp.zeros(output_layer_neurons_number)
        self.loss = 0
        if len(self.layers_layout) != self.number_of_layers:
            raise Exception('Number of specified node count should not exceed, or be less than the number of layers. '
                            'Expected a list of length {}, found {}!'.format(number_of_layers, len(self.layers_layout)))
        else:
            self.layers.append(Layer(cp.zeros(resolution), self.layers_layout[0]))
            for i in range(0, number_of_layers-1):
                self.layers.append(Layer(np.zeros(self.layers_layout[i]), self.layers_layout[i + 1]))

    def forward_propagation(self):
        for i in range(1, self.number_of_layers-1):
            self.layers[i].inputs = relu(self.layers[i-1].forward())
        self.outputs = soft_max(self.layers[self.number_of_layers - 1].forward())



    def back_propagation(self, batch_outputs, batch_layer_outputs, batch_layer_inputs, ytrain):
        m = ytrain.size
        self.loss = loss(batch_outputs, ytrain)
        d_losses = cp.array(self.loss)
        d_weights = cp.array((1 / m) * self.loss.dot(batch_layer_inputs[self.number_of_layers - 2]))
        d_biases = cp.array((1 / m) * cp.sum(self.loss))

        for i in range(self.number_of_layers - 1, 0, -1):
            d_losses.append(self.layers[i].weights.T.dot(d_losses[(self.number_of_layers - 1) - i]) * d_relu(np.transpose(batch_layer_outputs[i - 1])))

            d_weights.append((1 / m) * d_losses[self.number_of_layers - i].dot(batch_layer_inputs[i - 1]))

            d_biases.append((1 / m) * np.sum(d_losses[(self.number_of_layers - 1) - i]))

        return d_weights, d_biases

    def gradient_descent(self, batch_outputs, batch_layer_outputs, batch_layer_inputs, ytrain, learning_rate):
        d_weights, d_biases = self.back_propagation(batch_outputs, batch_layer_outputs, batch_layer_inputs, ytrain)
        for i in range(0, self.number_of_layers):
            self.layers[i].weights = self.layers[i].weights - learning_rate * d_weights[self.number_of_layers - 1 - i]
            self.layers[i].biases = self.layers[i].biases - learning_rate * d_biases[self.number_of_layers - 1 - i]

    def get_predictions(self):
        return cp.argmax(self.outputs, 0)

    def get_accuracy(self, y):
        print(self.get_predictions(), y)
        return cp.sum(self.get_predictions() == y) / y.size

    def train(self, xtrain, ytrain, iterations, learning_rate):
        batch_outputs = []
        batch_layer_inputs = cp.empty((self.number_of_layers, 0)).tolist()
        batch_layer_outputs = cp.empty((self.number_of_layers, 0)).tolist()
        for i in range(0, iterations):
            for j in range(0, len(xtrain)):
                self.layers[0].inputs = xtrain[j]
                self.forward_propagation()
                batch_outputs.append(self.outputs)
                for k in range(0, self.number_of_layers):
                    batch_layer_inputs[k].append(self.layers[k].inputs)
                    batch_layer_outputs[k].append(self.layers[k].forward())
            self.gradient_descent(batch_outputs, batch_layer_outputs, batch_layer_inputs, ytrain, learning_rate)
            batch_outputs = []
            batch_layer_inputs = cp.empty((self.number_of_layers, 0)).tolist()
            batch_layer_outputs = cp.empty((self.number_of_layers, 0)).tolist()
            if i % 10 == 0:
                print("Iteration: ", i)
                print("Accuracy: ", self.get_accuracy(ytrain))
