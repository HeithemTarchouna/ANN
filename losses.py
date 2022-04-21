import numpy as np


class Loss:
    def calculate(self, y_pred, y_true):
        # Calculate sample losses
        sample_losses = self.forward(y_pred, y_true)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


class SparseCategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        if len(y_true.shape) != 1:
            raise Exception("Avoid One-Hot Encoding or use CategoricalCrossentropy instead.")

        # number of samples in a batch
        samples = len(y_pred)
        # clip values between 1e-7 and 1 - 1e-7 to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # sparce categorical
        # confidence for the true target_class
        correct_confidences = y_pred_clipped[range(samples), y_true]
        neg_log_likelihoods = -np.log(correct_confidences)

        return neg_log_likelihoods


class CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            raise Exception("Target has to use One-Hot-Encoding.")
        # number of samples in a batch
        samples = len(y_pred)

        # clip values between 1e-7 and 1 - 1e-7 to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # confidence for the true target_class
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)

        return neg_log_likelihoods
