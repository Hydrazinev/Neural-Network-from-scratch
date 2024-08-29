import math
import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
np.random.seed(0)

class Layer_dense:
    def __init__(self,n_input,n_neuron) -> None:
        self.weight = np.random.randn(n_input,n_neuron)
        self.baises = np.zeros((1,n_neuron))
    def forward(self,input_data):
        self.output = np.dot(input_data,self.weight) + self.baises

class Activation_Relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probability = exp_values / np.sum(exp_values,axis=1,keepdims=True) 
        self.output = probability
class Loss:
    def calculate(self,output,target_value):
        sample_loss = self.forward(output,target_value)
        batch_loss = np.mean(sample_loss)
        return batch_loss
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        sample = len(y_pred)
        y_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidence = y_clipped[range(sample),y_true]
        elif len(y_pred.shape) == 2:
            correct_confidence = np.sum(y_clipped*y_true,axis=1)
        negetive_log_liklihood = -np.log(correct_confidence)
        return negetive_log_liklihood    

X,y = spiral_data(samples=100,classes=3)
dense1 =Layer_dense(2,3)
activation1 = Activation_Relu()

dense2 = Layer_dense(3,3)
activation2 = Activation_Softmax()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output,y)
print(loss)