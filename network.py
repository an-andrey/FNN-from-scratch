from layers import *
import utils

class FNN:
    def __init__(self, layers=None, gradient_method = "GD"):
        
        #if the user passes no layers, they can use the .add() method to add layers one by one
        if layers is None: 
            self.input_layer = None
            self.output_layer = None
            self.layers = []
            self.size = 0

        #otherwise, needs to pass at least 2 layers, 1 input and 1 output
        else: 
            e = TypeError("layers parameter expects a list of at least 2 layers. First one being an instance of class InputLayer, the rest of class Layer")

            if not isinstance(layers, list) or len(layers) < 2: 
                raise e
            
            for layer in layers[1:]:
                if isinstance(layer, Layer):
                    raise e
                

            self.input_layer = layers[0]
            self.layers = layers[1:]
            self.size = len(layers)

        supported_gradient_methods = ["GD"]

        if gradient_method not in supported_gradient_methods:
            raise ValueError(f"Provided gradient_method is invalid, use one of the supported methods: {supported_gradient_methods}")

        self.gradient_method = gradient_method

    def add(self, layer): # add another layer to list of layers
        
        if self.size == 0:
            if not isinstance(layer, InputLayer): 
                raise TypeError("first layer added needs to be an instance of InputLayer")
            
            self.input_layer = layer

        else:
            if not isinstance(layer, Layer): 
                raise TypeError("only instances of the Layer class can be added")
            
            self.layers.append(layer)

        self.size += 1

    # builds the network by making backward connections between the layers
    def build(self): 
        self.output_layer = self.layers[-1]
        self.layers[0].set_prev_layer(self.input_layer)

        for i in range(1, len(self.layers)): 
            self.layers[i].set_prev_layer(self.layers[i-1])
        

    # given a datapoint (x) makes a forward pass
    # returns the output of the output layer
    def __forward_pass(self, x):
        self.input_layer.a = x

        for layer in self.layers: 
            layer.compute_activations()
        
        return self.output_layer.a

    # once the forward pass is done, uses the target y to update the parameters of each layer
    def __backward_pass(self, y, lr):
        self.output_layer.error = utils.mse(self.output_layer.a, y) # computing MSE between fwd pass output and target

        for i in range(self.size - 1, -1, -1):
            self.layers[i].update_parameters(lr)

    def fit(self, X_train, y_train, lr): 
        if X_train.shape[1] != self.input_layer.size: 
            raise ValueError(f"Training set of dimension {X_train.shape} incompatible with input layer of size {self.input_layer.size}")

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Size of train set of dimension {X_train.shape} incompatible with target dimension {y_train.shape}")

        for i in range(X_train.shape[0]): 
            x = X_train[i, :]
            y = y_train[i, :]

            self.__forward_pass(x)
            self.__backward_pass(y)

    # Given a test set, computes the prediction of the network for every datapoint
    def predict(self, X_test):
        if X_test.shape[1] != self.input_layer.size: 
            raise ValueError(f"Test set of dimension {X_test.shape} incompatible with input layer of size {self.input_layer.size}")
        
        y_pred = np.zeros(shape=[X_test.shape[0], self.output_layer.size])
        
        for i in range(X_test.shape[0]): 
            x = X_test[i, :]
            y = X_test[i, :]
            
            y_pred[i,:] = self.__forward_pass(x)
        
        return y_pred
    

test = FNN()