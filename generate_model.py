from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
import csv
from dataclasses import dataclass


class Data:
    def __init__(self):
        self.temp_X = []
        self.temp_Y = []

    def add_row(self, x, y):
        self.temp_X.append(x)
        self.temp_Y.append([y])

    def make_arrays(self):
        self.X = np.array([*self.temp_X])
        self.Y = np.array([*self.temp_Y])


model_data = Data()

with open('./point-weighting-combined.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    useful_data = data[2:]
    for row in useful_data:
        result = [int(y) for y in row[1]]
        raw_data_row = [int(x) for x in row[2:]]
        model_data.add_row(raw_data_row, result)

model_data.make_arrays()

model = Sequential()
model.add(Dense(units=128, activation='sigmoid', input_dim=16))
model.add(Dense(units=1, activation='sigmoid'))

sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(model_data.X, model_data.Y, epochs=1500, verbose=False)

print(model.summary())
model.save('neural_network.model')
