import numpy as np
import utils

#Instead of creating each individual neurons, treating layers only
class Layer:
    def __init__(self, size, activation_fxn = utils.sigmoid, activation_fxn_dv = utils.sigmoid_dv):
        self.size = size
        self.b = np.random.random(size=self.size)
        self.w = None # constructed in the .build() method, once the previous layer is connected
        self.activation_fxn = activation_fxn
        self.activation_fxn_dv = activation_fxn_dv # Eventually could implement auto-diff, but manual inputs for now
        self.z = None # neurons' signals
        self.a = None # neurons' activations
        self.error = np.zeros(shape=size) # neurons' errors
    
    #Given the previous layer, creates the weight matrix
    def set_prev_layer(self, prev_layer):
        if not isinstance(prev_layer, Layer) or isinstance(prev_layer, InputLayer):
            raise TypeError("prev_layer object needs to be an instance of class Layer or InputLayer")

        self.prev_layer = prev_layer

        #since we know the size of the previous layer, we can also create the weight matrix
        self.w = np.random.random(size=[self.size, self.prev_layer.size])
        

    def compute_activations(self): #given activations of prev layer, computes the activations for the current layer
        self.z = self.w @ self.prev_layer.a + self.b 
        self.a = self.activation_fxn(self.z)

    # computes the gradients for every parameter, update weights and biases of the layer
    # and updates the error of the previous layer
    def update_parameters(self, lr):  

        if self.a is None or self.z is None: 
            raise ValueError("Make sure forward pass has been made")

        #computing the gradients and updating each weight and bias individually
        for j in range(self.size):
            
            #updating each bias of the current layer
            b_grad = 2*(self.a[j] - self.error[j])

            self.b[j] -= lr * b_grad

            #updating each weight for each neuron in the previous layer
            for k in range(self.w.shape[1]):
                # note, the partial dv of the signal wrt to the bias is 1, so the only thing different 
                # is the partial dv wrt to the weight 
                w_grad = b_grad*self.activation_fxn_dv(self.z[j]) 

                self.w[j][k] -= lr * w_grad

        #Finally, updating the error for the previous layer
        for k in range(self.prev_layer.error.shape[0]):

            error_grad = 0 

            for j in range(self.size): #the k_th activation of prev layer affects all neurons of currrent layer
                error_grad += 2*(self.a[j] - self.error[j])*self.activation_fxn_dv(self.z[j])*self.w[j][k]

            self.prev_layer.error[k] -= lr * error_grad

class InputLayer: 
    def __init__(self, size):
        self.size = size
        self.a = None # the activations for the input layer is simply the datapoint being passed into the network
        self.error = np.zeros(size=size)

