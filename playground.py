import numpy as np
from layers import Dense
import nnfs
from nnfs.datasets import vertical_data
import activations
import losses
import metrics
import matplotlib.pyplot as plt

nnfs.init()

# create a dataset
X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')

# helper variabless
epochs = 1000
best_weights_l1 = None
best_biases_l1 = None
best_weights_l2 = None
best_biases_l2 = None

best_loss = np.inf

# create the model
# create a dense layer with 2 input features and 3 output values
dense1 = Dense(2, 3, activation=activations.relu)
dense2 = Dense(3, 3, activation=activations.softmax)
# create the loss function
loss_function = losses.SparseCategoricalCrossentropy()


for cur_epoch in range(epochs):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # perform forward pass
    dense1.forward(X)
    dense2.forward(dense1.output)

    loss = loss_function.calculate(dense2.output, y)
    accuracy = metrics.Accuracy.result(dense2.output, y)
    if loss <= best_loss:
        print(f'new set of weights found : epoch {cur_epoch}')
        print(f'old loss = {best_loss}... new loss = {loss}')
        print(f'current accuracy {accuracy}')
        best_loss = loss
        best_weights_l1 = dense1.weights.copy()
        best_biases_l1 = dense1.biases.copy()
        best_weights_l2 = dense2.weights.copy()
        best_biases_l2 = dense2.biases.copy()
    else:
        dense1.weights = best_weights_l1.copy()
        dense1.biases = best_biases_l1.copy()
        dense2.weights = best_weights_l2.copy()
        dense2.biases = best_biases_l2.copy()



