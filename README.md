Example of construction of an Artificial Neural Network with the single-layer HurNet architecture for two-dimensional numerical matrices.

** ---> Click [here](https://github.com/sapiens-technology/SingleLayerHurNetCore) to use the CORE version with native Python that does not require prior installation of Numpy. <--- **

# Single Layer HurNet

This code is an algorithm designed, architected and developed by Sapiens Technology®️ and aims to build a simple Artificial Neural Network without hidden layers and with tabular data arranged in two-dimensional numerical matrices at the input and output. This is just a simplified example of the complete HurNet Neural Network architecture that can be used and distributed in a compiled form by Sapiens Technology®️ members in seminars and presentations. HurNet Neural Networks use a peculiar architecture (created by Ben-Hur Varriano) that eliminates the need for backpropagation in weights adjustment, making network training faster and more effective. In this example, we do not provide the network configuration structure for hidden layers and we reduce the dimensionality of the data to prevent potential market competitors from using our own technology against us. Although it is a simplified version of the HurNet network, it is still capable of assimilating simple patterns and some complex patterns that a conventional Multilayer Neural Network would not be able to abstract.

If you prefer, click [here](https://colab.research.google.com/drive/1l2moQTZmXBDDDrAZi_QW06h4JGGJOYry?usp=sharing) to run it via [Google Colab](https://colab.research.google.com/drive/1l2moQTZmXBDDDrAZi_QW06h4JGGJOYry?usp=sharing).

Click [here](https://zenodo.org/records/14048948) to read the full study.

## Installation

Before installing the main package, it is necessary to install the numpy package as a dependency.

```bash
pip install numpy
```

Or install the recommended version:

```bash
pip install numpy==1.25.2
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Single Layer HurNet.

```bash
pip install single-hurnet
```

## Usage
Basic usage example:
```python
# practical example teaching the neural network to add two values
from single_hurnet import SingleLayerHurNet # import main class
hurnet_neural_network = SingleLayerHurNet() # instantiation of the class object
# input examples arranged in a two-dimensional array
inputs = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
]
# an example output for each entry arranged in a two-dimensional array
outputs = [
    [3],
    [7],
    [11],
    [15]
]
# training the neural network with examples for the input patterns with the desired outputs
hurnet_neural_network.train( # performs training of the artificial neural network
    input_layer=inputs, # assignment of the input two-dimensional matrix for training
    output_layer=outputs, # assignment of the output two-dimensional matrix for training
    linear=False # if True it will recognize the patterns linearly, if False the patterns will be recognized non-linearly (default is False)
)
# test inputs with different values than those used in training
test_inputs = [
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]
# the network's output vectors must correspond to the sum of the input vectors, respecting the mathematical pattern of training
test_outputs = hurnet_neural_network.predict( # performs artificial neural network prediction with data inference
    input_layer=test_inputs # assignment of the input two-dimensional matrix to the prediction
)
# displays the test inputs and the outputs obtained for each of the inputs
print(f'Inputs: {test_inputs}') # displays the values to be predicted
print(f'Outputs: {test_outputs}') # displays the prediction result with the inference values
```

Note that the result is returned quickly and with 100% accuracy.

```bash
Inputs: [[2, 3], [4, 5], [6, 7], [8, 9]]
Outputs: [[5.0], [9.0], [13.0], [17.0]]
```
You can use as many numeric elements as you want in the input and output, as long as all inputs and outputs have the same amount of numbers.
```python
# in this example the pattern to be learned must correspond to two values, the first the sum of all the entries and the second the result of that sum multiplied by ten
from single_hurnet import SingleLayerHurNet # import main class
hurnet_neural_network = SingleLayerHurNet() # instantiation of the class object
# note that you can use as many numeric elements as you want in the input and output with whatever combination you want
# input examples arranged in a two-dimensional array
inputs = [ # all input vectors must have the same number of elements
    [1, 2, 1],
    [3, 4, 2],
    [5, 6, 3],
    [7, 8, 4]
]
# an example output for each entry arranged in a two-dimensional array
outputs = [ # all output vectors must have the same number of elements
    [4, 40],
    [9, 90],
    [14, 140],
    [19, 190]
]
# training the neural network with examples for the input patterns with the desired outputs
hurnet_neural_network.train( # performs training of the artificial neural network
    input_layer=inputs, # assignment of the input two-dimensional matrix for training
    output_layer=outputs, # assignment of the output two-dimensional matrix for training
    linear=False # if True it will recognize the patterns linearly, if False the patterns will be recognized non-linearly (default is False)
)
# test inputs with different values than those used in training
test_inputs = [
    [2, 3, 4],
    [4, 5, 3],
    [6, 7, 2],
    [8, 9, 1]
]
# the network's output vectors must correspond to the sum of the input vectors followed by the multiplication of this sum by ten (or values close to that), respecting the mathematical training pattern
test_outputs = hurnet_neural_network.predict( # performs artificial neural network prediction with data inference
    input_layer=test_inputs # assignment of the input two-dimensional matrix to the prediction
)
# displays the test inputs and the outputs obtained for each of the inputs
print(f'Inputs: {test_inputs}') # displays the values to be predicted
print(f'Outputs: {test_outputs}') # displays the prediction result with the inference values
# note that the result will be obtained extremely quickly and with 100% accuracy
```
```bash
Inputs: [[2, 3, 4], [4, 5, 3], [6, 7, 2], [8, 9, 1]]
Outputs: [[9.0, 90.0], [12.0, 120.0], [15.0, 150.0], [18.0, 180.0]]
```
Note that with the HurNet architecture it is possible to perform satisfactory learning with a reduced number of examples in the sample.
```python
# in this example we are making the network learn the pattern of the logical negation operator (not)
from single_hurnet import SingleLayerHurNet # import main class
hurnet_neural_network = SingleLayerHurNet() # instantiation of the class object
# realize that with just a few examples (two), satisfactory learning is already possible
inputs = [[0], [1]] # input samples
outputs = [[1], [0]] # output samples
# execution of neural network training
hurnet_neural_network.train(input_layer=inputs, output_layer=outputs)
# samples for neural network prediction
test_inputs = [[1], [0]]
# running neural network inference
test_outputs = hurnet_neural_network.predict(input_layer=test_inputs)
# displays the test inputs and the outputs obtained for each of the inputs
print(f'Inputs: {test_inputs}') # displays the values to be predicted
print(f'Outputs: {test_outputs}') # displays the prediction result with the inference values
# note that the result will be obtained extremely quickly and with 100% accuracy
# by the logic of the not operator, each output must be the polarity inversion of the corresponding input
```
```bash
Inputs: [[1], [0]]
Outputs: [[0.0], [1.0]]
```
Next, we will do an example where the neural network will learn the logic of the computational operator AND.
```python
# in this example we will make the network learn the logic of the and operator
from single_hurnet import SingleLayerHurNet
hurnet_neural_network = SingleLayerHurNet()
# for logical operators we can use 0 to represent false (off) and 1 to represent true (on)
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [0], [0], [1]]

hurnet_neural_network.train(input_layer=inputs, output_layer=outputs)

test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_outputs = hurnet_neural_network.predict(input_layer=test_inputs)

print(f'Inputs: {test_inputs}')
print(f'Outputs: {test_outputs}')
# in the and operator, we will only have true (one), when all inputs are true (equal to one)
```
```bash
Inputs: [[0, 0], [0, 1], [1, 0], [1, 1]]
Outputs: [[0.0], [0.0], [0.0], [1.0]]
```
Now see what a learning example for the logical operator OR would look like.
```python
# in this example we will make the network learn the logic of the or operator
from single_hurnet import SingleLayerHurNet
hurnet_neural_network = SingleLayerHurNet()

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [1]]

hurnet_neural_network.train(input_layer=inputs, output_layer=outputs)

test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_outputs = hurnet_neural_network.predict(input_layer=test_inputs)

print(f'Inputs: {test_inputs}')
print(f'Outputs: {test_outputs}')
# in the or operator, we will have true (one) when at least one of the inputs is true (equal to one)
```
```bash
Inputs: [[0, 0], [0, 1], [1, 0], [1, 1]]
Outputs: [[0.0], [1.0], [1.0], [1.0]]
```
Conventional neural networks that use backpropagation are incapable of learning the logic of the XOR operator without using hidden layers. But with the HurNet network this is completely possible at a practically instantaneous speed.
```python
# in this example we will teach the neural network to learn the logic of the xor operator, which is the most complex of the traditional logic gates.
from single_hurnet import SingleLayerHurNet
hurnet_neural_network = SingleLayerHurNet()

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

hurnet_neural_network.train(input_layer=inputs, output_layer=outputs)

test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_outputs = hurnet_neural_network.predict(input_layer=test_inputs)

print(f'Inputs: {test_inputs}')
print(f'Outputs: {test_outputs}')
# in the logical xor operator, we will have true (one) only when the two inputs are different (one equal to one and the other equal to zero)
# this computational problem remained unsolved for artificial neural network algorithms until the introduction of deep learning with hidden layers
# with Hurnet networks, learning the problematic xor operator becomes simple even with the absence of hidden layers in our network
```
```bash
Inputs: [[0, 0], [0, 1], [1, 0], [1, 1]]
Outputs: [[0.0], [1.0], [1.0], [0.0]]
```
To save a pre-trained model, simply call the saveModel function.
```python
from single_hurnet import SingleLayerHurNet
hurnet_neural_network = SingleLayerHurNet()

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

hurnet_neural_network.train(input_layer=inputs, output_layer=outputs)
# use the saveModel function to save a pre-trained model to the path defined in the name parameter "path"
hurnet_neural_network.saveModel(path='my_model.hur')
# the model will be saved in the local directory with the name my_model and the extension .hur, but you can choose the directory and name you want for saving
```
To load a pre-trained model, simply call the loadModel function.
```python
from single_hurnet import SingleLayerHurNet
hurnet_neural_network = SingleLayerHurNet()
# by loading a model that has already been trained and saved previously, no new training is necessary.
hurnet_neural_network.loadModel(path='my_model.hur')

test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_outputs = hurnet_neural_network.predict(input_layer=test_inputs)

print(f'Inputs: {test_inputs}')
print(f'Outputs: {test_outputs}')
```
```bash
Inputs: [[0, 0], [0, 1], [1, 0], [1, 1]]
Outputs: [[0.0], [1.0], [1.0], [0.0]]
```
Now we will make a comparison in the training and inference time of the HurNet network compared to conventional neural networks with backpropagation.
```python
# notice how the hurnet network code is much shorter and simpler
from time import time
start_time = time()

from single_hurnet import SingleLayerHurNet
hurnet_neural_network = SingleLayerHurNet()

inputs = [[5, 2], [4, 1], [5, 3], [3, 2]]
outputs = [[7], [5], [8], [5]]

hurnet_neural_network.train(input_layer=inputs, output_layer=outputs)

test_inputs = [[4, 2], [3, 1], [5, 4], [2, 1]]
test_outputs = hurnet_neural_network.predict(input_layer=test_inputs)

print(f'Inputs: {test_inputs}')
print(f'Outputs: {test_outputs}')

end_time = time()
duration_in_seconds =  end_time - start_time
print('time spent: '+str(duration_in_seconds))
# this example will run on a macbook m3 max (the most advanced of the moment)
```
```bash
Inputs: [[4, 2], [3, 1], [5, 4], [2, 1]]
Outputs: [[6.0], [4.0], [9.0], [3.0]]
time spent: 0.03168320655822754
```
Now we will do the same example with the TensorFlow code library. Note that it will require much more code and that the result will be much slower.
```bash
pip install tensorflow
```
```python
# note that with tensorflow the configuration is much more laborious, the code is larger and it takes much more time to complete the execution
from time import time
start_time = time()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = np.array([[5, 2], [4, 1], [5, 3], [3, 2]], dtype=float)
outputs = np.array([[7], [5], [8], [5]], dtype=float)

model = keras.Sequential([keras.Input(shape=(2,)), layers.Dense(4, activation='relu'), layers.Dense(1)])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(inputs, outputs, epochs=2000, verbose=0)

test_inputs = np.array([[4, 2], [3, 1], [5, 4], [2, 1]], dtype=float)
test_outputs = model.predict(test_inputs)

print(f'Inputs: {test_inputs.tolist()}')
print(f'Outputs: {test_outputs.tolist()}')

end_time = time()
duration_in_seconds =  end_time - start_time
print('time spent: '+str(duration_in_seconds))
# this example will run on a macbook m3 max (the most advanced of the moment)
```
Note that with TensorFlow it took 351 times more time (11.107148885726929 divided by 0.03168320655822754) than we would have spent with the HurNet network and yet the TensorFlow result was obtained with a much lower level of accuracy. Also remember that this time can increase exponentially as more data and more layers are added to the network.
```bash
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
Inputs: [[4.0, 2.0], [3.0, 1.0], [5.0, 4.0], [2.0, 1.0]]
Outputs: [[6.13881778717041], [4.4135260581970215], [8.73843765258789], [3.6166493892669678]]
time spent: 11.107148885726929
```
Now look at the same example using the PyTorch code library.
```bash
pip install torch
```
```python
# note that with pytorch the configuration is much more laborious, the code is larger and it takes much more time to complete the execution
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

start_time = time()

inputs = np.array([[5, 2], [4, 1], [5, 3], [3, 2]], dtype=float)
outputs = np.array([[7], [5], [8], [5]], dtype=float)
inputs_tensor = torch.from_numpy(inputs).float()
outputs_tensor = torch.from_numpy(outputs).float()

model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 2000
for _ in range(epochs):
    predictions = model(inputs_tensor)
    loss = criterion(predictions, outputs_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_inputs = np.array([[4, 2], [3, 1], [5, 4], [2, 1]], dtype=float)
test_inputs_tensor = torch.from_numpy(test_inputs).float()
model.eval()
with torch.no_grad():
    test_outputs_tensor = model(test_inputs_tensor)
    test_outputs = test_outputs_tensor.numpy()

print(f'Inputs: {test_inputs.tolist()}')
print(f'Outputs: {test_outputs.tolist()}')

end_time = time()
duration_in_seconds =  end_time - start_time
print('time spent: '+str(duration_in_seconds))
# this example will run on a macbook m3 max (the most advanced of the moment)
```
With PyTorch optimizations it still took 13 times more time than with HurNet.
```bash
Inputs: [[4.0, 2.0], [3.0, 1.0], [5.0, 4.0], [2.0, 1.0]]
Outputs: [[6.003105640411377], [4.010413646697998], [8.996774673461914], [3.0186996459960938]]
time spent: 0.40598297119140625
```

## Methods
### saveModel (function return type: bool): Returns True if the training save is successful or False otherwise
Parameters
| Name          | Description                                                       | Type | Default Value |
|---------------|-------------------------------------------------------------------|------|---------------|
| path          | path and name of the generated file for the saved training model  | str  | ''            |

### loadModel (function return type: bool): Returns True if the training load is successful or False otherwise
Parameters
| Name          | Description                                                       | Type | Default Value |
|---------------|-------------------------------------------------------------------|------|---------------|
| path          | path and name of the model training file to be loaded             | str  | ''            |

### train (function return type: bool): Returns True if training is successful or False otherwise
Parameters
| Name          | Description                                                       | Type | Default Value |
|---------------|-------------------------------------------------------------------|------|---------------|
| input_layer   | input two-dimensional array                                       | list | []            |
| output_layer  | output two-dimensional array                                      | list | []            |
| linear        | True for linear calculation or False for non-linear calculation   | bool | False         |

### predict (function return type: list): Returns a two-dimensional array with the inference results
Parameters
| Name          | Description                                                       | Type | Default Value |
|---------------|-------------------------------------------------------------------|------|---------------|
| input_layer   | input two-dimensional array                                       | list | []            |

Check out an example below with all the features available in the current package.
```python
# example to teach the network to calculate double a number
from single_hurnet import SingleLayerHurNet # import main class
hurnet_neural_network = SingleLayerHurNet() # instantiation of the class object
# network training dataset
inputs = [[2], [4], [6], [8]] # input samples
outputs = [[4], [8], [12], [16]] # output samples
linear = True # inference will be faster with linearity enabled
# dataset for test prediction
test_inputs = [[1], [3], [5], [7]] # input data for inference
test_outputs = [] # variable to store the prediction response
if hurnet_neural_network.train(input_layer=inputs, output_layer=outputs, linear=linear): # checks whether the training was successful
    if hurnet_neural_network.saveModel(path='model_for_double'): # checks if the model was saved correctly (the .hur extension is optional)
        new_neural_network = SingleLayerHurNet() # creates a new in-memory network to test model loading without training the current network
        if new_neural_network.loadModel(path='model_for_double'): # checks if the model was loaded correctly (the .hur extension is optional)
            test_outputs = new_neural_network.predict(input_layer=test_inputs) # returns the prediction of the inference
# finalizes the execution of the model by presenting the data
if len(test_outputs) > 0: # checks if there is data in the prediction
    print(f'Inputs: {test_inputs}') # displays the data to be predicted
    print(f'Outputs: {test_outputs}') # displays the prediction result
else: print('Training was unsuccessful and inference could not be performed.') # displays a failure message if there is no prediction
# this code contains all the features available in the single layer hurnet algorithm
# the single layer hurnet algorithm uses only the input layer to predict the output layer
```
```bash
Inputs: [[1], [3], [5], [7]]
Outputs: [[2.0], [6.0], [10.0], [14.0]]
```

## Contributing

We do not accept contributions that may result in changing the original code.

Make sure you are using the appropriate version.

## License

This is proprietary software and its alteration and/or distribution without the developer's authorization is not permitted.

<sub>Varriano, B.-H., & Sapiens Technology®. (2024). Single Layer HurNet: A Neural Network Without Backpropagation for Two-Dimensional Matrices (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.14048948</sub>
