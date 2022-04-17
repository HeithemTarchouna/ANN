from layer_dense import LayerDense
from model import Model
import math
import pandas as pd


def main():
    url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
    c = pd.read_csv(url)

    model = Model()
    model.add(LayerDense(3, activation_function=math.tanh))
    model.add(LayerDense(2, activation_function=math.tanh))
    model.add(LayerDense(1))
    model.build(c)
    model.summary()


if __name__ == "__main__":
    main()
