# -*- coding: utf-8 -*-
"""國立臺北科技大學_類神經網路_連續型活化函數(20230406).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J_P58HPajs1e4ybckGXLQ-px13iWcsao
"""

import numpy as np

inputs = [] # x
inputs.append(np.array([0.922]))
inputs.append(np.array([0.459]))
inputs.append(np.array([0.984]))
inputs.append(np.array([0.794]))
inputs.append(np.array([0.119]))
inputs.append(np.array([0.258]))
inputs.append(np.array([0.734]))
inputs.append(np.array([0.123]))
inputs.append(np.array([0.713]))
inputs.append(np.array([0.943]))
labels = np.array([0.559, 0.298, 0.639, 0.516, 0.077, 0.167, 0.477, 0.079, 0.463, 0.612]) # y
Iters = 10
no_of_inputs = 1
np.random.seed(55)
weights = np.random.randn(no_of_inputs)
print("initial: " + str(weights))
learning_rate = 0.95

# Commented out IPython magic to ensure Python compatibility.
Err = []
#_W = []
for _ in range(Iters): 
  err = 0
  W = []
  for _input, label in zip(inputs, labels): 
    predicted = np.dot(_input, weights) # dot product 
    weights -= learning_rate * (label - predicted) * (-1) * _input
    err += (label - predicted) ** 2
    #W.append(weights)
  Err.append(err/len(inputs))
  #_W.append(np.std(W))
print("trained: " + str(weights))
import matplotlib.pyplot as plt
# %pylab inline
plt.plot(range(0, len(Err)),Err)
plt.xlabel('Iteration')
plt.ylabel('loss/error')

