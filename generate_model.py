from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
import csv
from dataclasses import dataclass

def open_csv():
    with open('./point-weighting-combined-new.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    return data


def get_data_Y():
    data = open_csv()

    useful_data = data[2:]

    new = []

    buff = 0
    index = 0
    while buff < len(useful_data):
        a = useful_data[buff + index][1]

        new.append(np.array([int(a)]))
        buff += 2

    return np.array([*new])



def get_data_X():
    data = open_csv()

    useful_data = data[2:]

    new = []

    buff = 0
    index = 0
    while buff < len(useful_data):
        a = useful_data[buff + index]
        b = useful_data[buff + index + 1]

        total = []
        for it_a, it_b in zip(a[2:], b[2:]):
            total.append(int(it_a) - int(it_b))

        new.append(np.array(total))
        buff += 2

    return np.array([*new])


if __name__ == "__main__":

    Y = get_data_Y()
    X = get_data_X()

    print(Y)
    print(len(Y))

    print(X)
    print(len(X))

    model = Sequential()
    model.add(Dense(units=64, activation='sigmoid', input_dim=16))
    model.add(Dense(units=1, activation='sigmoid'))

    sgd = optimizers.SGD(lr=1)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(X, Y, epochs=1500, verbose=False)

    print(model.summary())
    model.save('neural_network.model')
