import numpy as np


class Model:
    def __init__(self, layers=None):
        if layers is None:
            self.layers = np.array([])
        else:
            self.layers = layers

    def add(self, layer):
        self.layers = np.append(self.layers, layer)

    def build(self, X):
        # Fixing the input shape of the first last in case the user didn't specify it explicitly
        self.layers[0].input_shape = X.shape[1]

        # Initialising the weights and biases
        for l in range(len(self.layers)):
            if l != 0:
                self.layers[l].input_shape = self.layers[l - 1].number_of_neurons
            self.layers[l].weights = np.random.random_sample(
                (self.layers[l].number_of_neurons, self.layers[l].input_shape))
            self.layers[l].biases = np.vstack([np.ones(self.layers[l].number_of_neurons)] * X.shape[0]).T

    def summary(self):
        print("Model Name : 1")
        print("_________________________________________________________________")
        print(" Layer (type)                Output Shape              Param #   ")
        print("=================================================================")
        l = 0
        for layer in self.layers:
            print(
                f" Layer{l}                      {(1, layer.number_of_neurons)}                    {layer.weights.shape[0] * layer.weights.shape[1] + layer.biases.shape[0]}        ")
            l = l + 1

    def __feedForward(self, X):
        self.layers[0].inputs = X.T
        for l in range(0, len(self.layers)):
            self.layers[l].outputs = self.layers[l].activation_function(
                np.dot(self.layers[l].weights, self.layers[l].inputs)) + self.layers[l].biases
            if l + 1 <= len(self.layers) - 1:
                self.layers[l + 1].inputs = self.layers[l].outputs

    def fit(self, X, y=None, epochs=None):
        self.__feedForward(X)
