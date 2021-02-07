from tensorflow import keras
import numpy as np
import csv

from generate_model import get_data_Y, get_data_X


# Import the model made by ./generate_model.py
model = keras.models.load_model('neural_network.model')

# Gets the data from the model generator file "./generate_model.py"
X = get_data_X()
Y = get_data_Y()


def test_each_match():
    # Get the result actual win value, and the one predicted by the model
    for x, y in zip(X, Y):
        # Predicts the Y value from a given X
        result = model.predict(np.array([x]))
        # Prints the values in the (actual, predict) format
        print(y[0], result[0][0])


def get_weights_method_one():
    """The first method for getting the weights"""
    first_layer_weights = model.layers[0].get_weights()[0]
    print(first_layer_weights)


def get_weights_method_two():
    """The second method for getting the weights"""
    for layer in model.layers:
        print(layer.get_config(), layer.get_weights())


if __name__ == "__main__":
    test_each_match()
