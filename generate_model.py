from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
import csv
from typing import List


def open_csv() -> List:
    """This opens the data from a csv and returns a list of it as data"""
    with open('./point-weighting-combined-new.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    return data


def get_data_Y():
    """Gets the value of A winning, this is the second column in the csv for
    each match to use the Y value"""
    data = open_csv()

    # Get the parts of the data that are not the team (A, B) and the value
    # showing whether or not they lost. Everything after the second column
    useful_data = data[2:]

    # The final list of Y values that will be turned into a numpy array and
    # returned later in this script
    final = []

    # Sets the buffer location, if increased by two after the loop, it will go
    # through each row two at a time
    buff = 0
    # Goes through each of the rows until it is at the limit
    while buff < len(useful_data):
        # Gets the value that shows if A has won the match
        a = useful_data[buff][1]

        # Adds the A values as a numpy array to the final list
        final.append(np.array([int(a)]))
        # Increases the buffer, goes to the next match
        buff += 2

    # Returns the final list as a numpy array
    return np.array([*final])



def get_data_X():
    """Gets the rest of the data in each match and find the difference in the
    two teams for each match and save that as a numpy array"""
    data = open_csv()

    # Get all the columns after the initial team letter value and the value
    # that shows if A has won the match
    useful_data = data[2:]

    # Sets the final list to be returned as the X values
    final = []

    # Sets the location to read from in the list of matches
    buff = 0

    # Loops through each match
    while buff < len(useful_data):
        # Get all the fields for the team A after the first two columns
        a = useful_data[buff]
        # Get all the fields for the team B after the first two columns
        b = useful_data[buff + 1]

        # Each total list is the difference between A and B score per row
        total = []
        # Get the totals of the difference of each column after the second
        for it_a, it_b in zip(a[2:], b[2:]):
            # Add the difference to the list called total to relate the team A
            # score with the team B score to better understand what winning a
            # match would look like on a team by team basis
            total.append(int(it_a) - int(it_b))

        # Add each total list (the difference between A and B score per row) to
        # the final list to have the total metrics for each match
        final.append(np.array(total))
        # Increases the buffer through the list
        buff += 2

    return np.array([*final])


if __name__ == "__main__":

    # Gets the value for X and Y
    Y = get_data_Y()
    X = get_data_X()

    # Sets up the model and it's structure
    model = Sequential()
    # Adds layers, the input and middle layer and specifies it's activation
    # function is the sigmoid function. It also specifies the input dimension
    # if 16, this means we are looking at 16 independent variables per match
    # and we have 64 matches, the match data consists of the difference of the
    # score of team A and Team B for each one of the sixteen variables.
    model.add(Dense(units=64, activation='sigmoid', input_dim=16))
    model.add(Dense(units=1, activation='sigmoid'))

    # This sets up the optimizer
    sgd = optimizers.SGD(lr=1)
    # This runs the specifies optimizer with the type of loss
    model.compile(loss='mean_squared_error', optimizer=sgd)

    # This fits the data of X to the Y output values
    model.fit(X, Y, epochs=1500, verbose=False)

    # This shows the summary of the model, the layers, shape, and parameters
    print(model.summary())
    # This saves the model so it can be used without having to retrain the it
    model.save('neural_network.model')
