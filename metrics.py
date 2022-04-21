import numpy as np


class Accuracy:
    @staticmethod
    def calculate(y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_true)
        return accuracy
