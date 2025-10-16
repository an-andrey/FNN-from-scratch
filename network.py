import numpy as np
import utils

#Instead of creating each individual neurons, treating layers only
class Layer:
    def __init__(self, size = 1, activation_fxn = utils.sigmoid, activation_fxn_dv = utils.sigmoid_dv):
        self.size = size
        self.b = np.random.random(size=[self.size, 1])
        # to be defined by the network, since you need to know the size of prev-layer
        self.w = None
        self.prev_layer_size = None
        self.activation_fxn = activation_fxn
        self.activation_fxn_dv = activation_fxn_dv
    
    def set_weights(self, in_size):
        self.prev_layer_size = in_size
        self.w = np.random.random(size=[self.size, self.prev_layer_size])
        

    def compute_activation(self, x):
        try:
            if x.ndim == 1: 
                x = x[:, None]
            
            if not self.w: 
                raise ValueError("Make sure that the weights are initialized")
            
            activation = self.activation_fxn(self.w @ x + self.b)
            return activation

        except TypeError:  
            raise TypeError(f"x must an numpy array") 

        except ValueError:
            raise ValueError(f"x is of incorrect shape, expected: {[self.prev_layer_size,1]} got {x.shape}")
        
    
    # update weights and biases of the layer
    # returns the gradient for the activation for the previous layer
    def backprop(self, dell, lr): #dell is the gradient coming from the next layer
        prev_w = self.w
        prev_b = self.b

        