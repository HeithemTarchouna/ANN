import numpy as np
from layers import Dense
import nnfs
from nnfs.datasets import spiral_data
import activations
import losses
import metrics

nnfs.init()

# create a dataset
X, y = spiral_data(samples=100, classes=3)

# create a dense layer with 2 input features and 3 output values
dense1 = Dense(2, 3, activation=activations.relu)

# perform forward pass
dense1.forward(X)

dense2 = Dense(3, 3, activation=activations.softmax)

dense2.forward(dense1.output)

loss_function = losses.SparseCategoricalCrossentropy()
loss = loss_function.calculate(dense2.output, y)
print(f"loss:{loss}")
print(metrics.Accuracy.calculate(dense2.output, y))




