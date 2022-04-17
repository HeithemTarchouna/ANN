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
            print((self.layers[l].number_of_neurons, self.layers[l].input_shape))
            self.layers[l].weights = np.random.random_sample(
                (self.layers[l].number_of_neurons, self.layers[l].input_shape))
            self.layers[l].biases = np.ones(self.layers[l].number_of_neurons)

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

    def fit(self, X, y, epochs):
        pass
