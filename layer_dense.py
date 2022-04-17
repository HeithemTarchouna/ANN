class LayerDense:
    def __init__(self, number_of_neurons, activation_function=None, input_shape=None):
        self.input_shape = input_shape
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function
        self.weights = None
        self.biases = None
