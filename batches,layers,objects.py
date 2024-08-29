import numpy as np
np.random.seed(0)

input_data = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

class Layers:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  
    def forward(self, input_data):
        self.output = np.dot(input_data, self.weights) + self.biases 



layer1  = Layers(4, 5) #number of inputs,number of neurons
layer2 = Layers(5, 2)

layer1.forward(input_data)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


