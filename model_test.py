from tensorflow import keras
import numpy as np
import csv

from generate_model import get_data_Y, get_data_X


# Import the model made by ./generate_model.py
model = keras.models.load_model('neural_network.model')

X = get_data_X()
Y = get_data_Y()

# Get the result and print it
for x, y in zip(X, Y):
    result = model.predict(np.array([x]))
    # print(result[0][0])
    print(y[0], result[0][0])
