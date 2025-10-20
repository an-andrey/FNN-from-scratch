import numpy as np


sigmoid = lambda x : 1/(1 + np.exp(-x))

sigmoid_dv = lambda x : np.exp(-x)/(1+np.exp(-x))**2

mse = lambda x, y : np.mean((x - y)**2)

relu = lambda x : np.maximum(np.zeros(shape=x.shape), x)

relu_dv = lambda x: (x > 0).astype(float)

identity = lambda x: x
identity_dv = lambda x: 1