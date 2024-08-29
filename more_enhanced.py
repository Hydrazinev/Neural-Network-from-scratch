import numpy as np
input = [1,2,3,2,5]

weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]

layer_outputs = []

for neuron_weight,newron_bias in zip(weights,biases):
    output_neuron = 0
    for n_input, weight in zip(input,neuron_weight):
        output_neuron += n_input*weight
    output_neuron += newron_bias
    layer_outputs.append(output_neuron)
print(layer_outputs)

output1 = np.dot(weight,input) + biases
print(output1)   