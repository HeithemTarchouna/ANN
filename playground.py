import numpy as np

import layers
from layers import Dense
import nnfs
from nnfs.datasets import vertical_data
import activations
import losses
import metrics
import matplotlib.pyplot as plt

# nnfs.init()
#
# # Create dataset
# X, y = vertical_data(samples=100, classes=3)
# # Create model
# dense1 = layers.Dense(2, 3, activation=activations.relu)  # first dense layer, 2 inputs
# dense2 = layers.Dense(3, 3, activation=activations.softmax)  # second dense layer, 3 inputs, 3 outputs
# loss_function = losses.SparseCategoricalCrossentropy()
#
# # Helper variables
# lowest_loss = float("inf")  # some initial value
# best_dense1_weights = dense1.weights.copy()
# best_dense1_biases = dense1.biases.copy()
# best_dense2_weights = dense2.weights.copy()
# best_dense2_biases = dense2.biases.copy()
# epochs = 1000
#
# for epoch_i in range(epochs):
#     dense1.forward(X)
#     dense2.forward(dense1.output)
#     loss_value = loss_function.calculate(dense2.output, y)
#     if loss_value < lowest_loss:
#         lowest_loss = loss_value
#         # print(f"New weights found at epoch {epoch_i}")
#         # print(f"New lowest loss =  {lowest_loss}")
#         # # print(f"Current Accuracy = {metrics.Accuracy().result(dense2.output, y)  }")
#         # print(f"-------------------------------------------")
#
#         best_dense1_weights = dense1.weights.copy()
#         best_dense1_biases = dense1.biases.copy()
#         best_dense2_weights = dense2.weights.copy()
#         best_dense2_biases = dense2.biases.copy()
#     else:
#         dense1.weights = best_dense1_weights.copy()
#         dense2.weights = best_dense2_weights.copy()
#         dense1.biases = best_dense1_biases.copy()
#         dense2.biases = best_dense2_biases.copy()
#
#         if epoch_i < epochs - 1:
#             # generate new weights
#             dense1.weights += 0.05 * np.random.randn(2, 3)
#             dense1.biases += 0.05 * np.random.randn(1, 3)
#             dense2.weights += 0.05 * np.random.randn(3, 3)
#             dense2.biases += 0.05 * np.random.randn(1, 3)
# print("Best Results")
# print(f"Lowest loss = {lowest_loss}")
#
# dense1.forward(X)
# dense2.forward(dense1.output)
# loss_value = loss_function.calculate(dense2.output, y)
# print(metrics.Accuracy.result(y_pred=dense2.output, y_true=y))


x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2, 0]  # weights
b = 1.0  # bias

# multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# adding weighted inputs together and a bias
z = xw0 + xw1 + xw2 + b

# relu activation function
y = activations.relu(z)

# Backward pass

# The derivative from the next layer
dvalue = 1.0

# Derivative of Relu and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)

# Partial derivates of the summation , the chain rule : wrt weighted inputs
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1

drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2

# Partial derivates : the chain rule : wrt bias
dsum_db = 1
drelu_db = drelu_dz * dsum_db
# Partial derivates of the multiplication , the chain rule : wrt to inputs
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
print("loss wrt inputs")
print(drelu_dx0)
print(drelu_dx1)
print(drelu_dx2)

# Partial derivates of the multiplication , the chain rule : wrt to weights
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2
print("loss wrt weight")
print(drelu_dw0)
print(drelu_dw1)
print(drelu_dw2)
