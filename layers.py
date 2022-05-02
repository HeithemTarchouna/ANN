import numpy as np
import activations


class Dense:
    def __init__(self, n_inputs, n_neurons, activation=activations.linear):
        """input matrix is in the form of             [ _X1_ ] :First sample : [ feature:1, .. ,feature:n ]
                                                   [ _X2_ ] :Second sample
                                                   [ _Xi_ ] :i-th sample
                                                   [ _XN_ ] :Last sample
        shape of input matrix is : N-samples x n_inputs
        -------------------------------------------------------------------------------------------------------------
        weights matrix is used here in the form of [ | , | , .. , |         ] Wi as row vector
                                                   [ W1, W2, .. , Wn_neurons]
                                                   [ | , | , .. , |         ]
        shape of weight matrix is : n_inputs x n_neuron
        -------------------------------------------------------------------------------------------------------------
        output = X @ W + B (here W is already transposed so we don't have to do it everytime)
        when we perform feedforward we get [ _output of layer for the sample 1_ ]  : (row vector : 1 x n_neurons)
                                           [ _output of layer for the sample 2_ ]  : (row vector : 1 x n_neurons)
                                           [ _output of layer for the sample N_ ]  : (row vector : 1 x n_neurons)
        shape of output is : N-samples X n_neurons
        -------------------------------------------------------------------------------------------------------------
        biases is of the shape : 1 x n_neurons so it's easier to add it to the result of the dot product
        without any additional operation like transposing etc
       """

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None
        self.activation = activation


    def forward(self, inputs):
        # calculate output values from inputs, weights and biases
        self.output = self.activation(np.dot(inputs, self.weights) + self.biases)
