#  Game Design Scoring Regression Model


### Data
The data that this model used to train is the csv `./point-weighting-combined.csv` and was made in a spreedsheet

### Building the model
Run the script `./generate_model.py` to build the model

### Usage
To run the model that you built, this is all the code needed

```py
from tensorflow import keras  
import numpy as np  
  
# Import the model made by ./generate_model.py  
model = keras.models.load_model('neural_network.model')       
                                                              
# Give it match data to predict
to_predict = np.array([5, 0, 1, 1, 1, 1, 4, 3, 4, 0, 3, 3, 3, 3, 3, 6])
                                                              
# Get the result and print it
result = model.predict(np.array([to_predict]))                
print(result)                       
```
