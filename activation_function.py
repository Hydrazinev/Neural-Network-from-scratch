import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]] 

X,y = spiral_data(100,3)
class Layers:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # Corrected this line
    def forward(self, input_data):
        self.output = np.dot(input_data, self.weights) + self.biases 
class ActivationRelu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
layer1  = Layers(2, 5) #number of inputs,number of neurons
activation_1 = ActivationRelu()

layer1.forward(X)
print(layer1.output)
print('--------------------------------')
activation_1.forward(layer1.output)
print(activation_1.output)