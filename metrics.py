import numpy as np


class Accuracy:
    def result(y_pred, y_true):
        predictions = y_pred
        # check if it's in hot-one encoding and convert back to sparce if true
        if len(y_pred.shape) == 2:
            predictions = np.argmax(y_pred, axis=1)

        accuracy = np.mean(predictions == y_true)
        return accuracy
