import numpy as np


sigmoid = lambda x : 1/(1 + np.exp(-x))

sigmoid_dv = lambda x : np.exp(-x)/(1+np.exp(-x))**2