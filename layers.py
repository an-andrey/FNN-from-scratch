import numpy as np
import utils

#Instead of creating each individual neurons, treating layers only
class Layer:
    def __init__(self, size, activation_fxn = utils.sigmoid, activation_fxn_dv = None):
        self.size = size
        self.b = np.zeros(shape=self.size)
        self.w = None # constructed in the .build() method, once the previous layer is connected
        self.activation_fxn = activation_fxn
        self.activation_fxn_dv = None

        if self.activation_fxn is utils.sigmoid: 
            self.activation_fxn_dv = utils.sigmoid_dv
        
        elif self.activation_fxn is utils.relu: 
            self.activation_fxn_dv = utils.relu_dv
        
        elif activation_fxn_dv is None: 
            raise ValueError("If the provided activation function isn't part of the supported functions, please provided the activation function derivative")

        else: 
            self.activation_fxn_dv = activation_fxn_dv

        self.z = None # neurons' signals
        self.a = None # neurons' activations
        self.error = np.zeros(shape=size) # neurons' errors
        self.is_output_layer = False
    
    def set_output_layer(self):
        self.is_output_layer = True

    #Given the previous layer, creates the weight matrix
    def set_prev_layer(self, prev_layer):
        if not isinstance(prev_layer, Layer) and not isinstance(prev_layer, InputLayer):
            raise TypeError(f"prev_layer object needs to be an instance of class Layer or InputLayer, got {type(prev_layer)}")

        self.prev_layer = prev_layer

        #since we know the size of the previous layer, we can also create the weight matrix
        self.w = np.random.randn(self.size, prev_layer.size)*0.1
        

    def compute_activations(self): #given activations of prev layer, computes the activations for the current layer
        self.z = self.w @ self.prev_layer.a + self.b 
        self.a = self.activation_fxn(self.z)

    # computes the gradients for every parameter, update weights and biases of the layer
    # and updates the error of the previous layer
    def update_parameters(self, lr):  

        if self.a is None or self.z is None: 
            raise ValueError("Make sure forward pass has been made")

        #computing the gradients and updating each weight and bias individually
        self.b -= lr * self.error
        #the outer product creates a matrix, where every row i = self.error[i] * self.prev_layer.a
        self.w -= lr * np.outer(self.error, self.prev_layer.a) 

        #Finally, updating the error for the previous layer - if the previous one isn't the input layer
        if not isinstance(self.prev_layer, InputLayer):
            self.prev_layer.error = self.prev_layer.activation_fxn_dv(self.prev_layer.z) * (self.w.T @ self.error) # k x 1 * (  k x j @ j x 1 ) -> k x 1

class InputLayer: 
    def __init__(self, size):
        self.size = size
        self.a = None # the activations for the input layer is simply the datapoint being passed into the network
        self.error = np.zeros(shape=size)

