from tensorflow import keras
import numpy as np

# Import the model made by ./generate_model.py
model = keras.models.load_model('neural_network.model')

# Give it match data to predict
to_predict = np.array([5, 0, 1, 1, 1, 1, 4, 3, 4, 0, 3, 3, 3, 3, 3, 6])

# Get the result and print it
result = model.predict(np.array([to_predict]))
print(result)
