import numpy as np

input = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]  # Ensure all sub-arrays have the same length
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

weights1 = [[0.4, 0.5, -0.1], [0.6, -0.53, 0.54], [-0.5, -0.27, 0.17]]
biases2 = [4,0.23,-0.32]
output1 = np.dot(input,np.array(weights).T) + biases
output2 = np.dot(output1,np.array(weights1).T) + biases2
print(output1)
print(output2)
