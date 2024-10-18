import numpy as np
import keras

class BuildAModel:
    def __init__(self,input_shape:tuple,output_shape,activation:str,hidden_layer_dims:list,output_activation:str|None=None):
        """
        Initialize model parameters
        Args:
            input_shape: training data shape
            activation: activation function for hidden layers
            hidden_layer_dims: hidden layer dimensions in the form of a list, where each element is the number of neurons
        """
        self.input_shape = input_shape
        self.activation = activation
        self.output_shape = output_shape
        self.hidden_layer_dims = hidden_layer_dims
        self.output_activation = output_activation
        self.model = None
    
    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = keras.layers.Dense(self.hidden_layer_dims[0],activation=self.activation)(inputs)
        if len(self.hidden_layer_dims) > 1:
            for dim in self.hidden_layer_dims[1:]:
                x = keras.layers.Dense(dim,activation=self.activation)(x)
        outputs = keras.layers.Dense(self.output_shape,activation=self.output_activation)(x)
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        return self.model