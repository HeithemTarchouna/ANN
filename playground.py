# Single neuron
import numpy as np
from layers import Dense
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# create a dataset
X, y = spiral_data(samples=100, classes=3)

# create a dense layer with 2 input features and 3 output values
dense1 = Dense(2, 3)

# perform forward pass
dense1.forward(X)

# print the output for some samples
print(dense1.output[:5])
