from layer_dense import LayerDense
from model import Model
import math


def main():
    model = Model(
        [LayerDense(3, activation_function=math.tanh, input_shape=[1]),
         LayerDense(2, activation_function=math.tanh),
         LayerDense(1)])


if __name__ == "__main__":
    main()
