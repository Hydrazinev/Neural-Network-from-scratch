
#INPUT->EXPONTIATE->NORMALIZATION->OUTPUT

import math
import numpy as np
import nnfs
layers_output = [4.8,1.21,2.385]
euler = math.e
exp_values = []

for values in layers_output:
    exp_values.append(euler**values)

print(exp_values)
 
norm_base = sum(exp_values)
norm_values = []

for val in exp_values:
    norm_values.append(val/norm_base)

print(norm_values)
##annother short  method
nnfs.init()
ano_layers_output = [4.8,1.21,2.385]
ano_exp_values = np.exp(ano_layers_output)
ano_norm_values = ano_exp_values/sum(ano_exp_values)

print(ano_exp_values)
print(ano_norm_values)
