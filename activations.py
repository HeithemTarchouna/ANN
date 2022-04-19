import numpy as np


def relu(inputs):
    # if element >0 return element
    # else return 0
    return np.maximum(0, inputs)


def linear(inputs):
    return inputs


def softmax(inputs):
    # inputs are usually the output layer neurons values
    # the reason why we substract the max value of output layer values is to prevent very large values
    # (positive large values can cause the exponent function to explode)
    # if we substract the largest value, then the values will range between -infinity and 0
    # therefore when we apply the exp fonction the values are between 0 and 1 => no large values
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

    # Now normalize values
    # thanks to the normalisation the proportions will not change therefore the outputs won't change
    # even after substracting the max value
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return probabilities
