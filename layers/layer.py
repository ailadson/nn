import config
from functions.activations import *

class Layer:
    def __init__(self, prev_layer, output_shape, activation_func_name):
        self.prev_output_shape = prev_layer.output_shape if prev_layer else None
        self.output_shape = output_shape

        activation_funcs = get_activation_functions(activation_func_name)
        self.activation_func = activation_funcs[0]
        self.deriv_activation_func = activation_funcs[1]

        self.prev_layer = prev_layer
        self.next_layer = None
        if prev_layer is not None:
            prev_layer.next_layer = self

        self.z_output = np.zeros(output_shape, dtype=config.FLOAT_TYPE)
        self.output = np.zeros(output_shape, dtype=config.FLOAT_TYPE)

    def back_propagate(self):
        raise Exception(f"Not Implemented for {self}!")

    def forward_propagate(self):
        raise Exception(f"Not Implemented for {self}!")

    def has_weights(self):
        raise Exception(f"Not Implemented for {self}!")

def get_activation_functions(name):
    if name == 'id':
        activation_func = lambda val, des: np.copyto(des, val)
        deriv_activation_func = lambda val, des: des.fill(1)
    elif name == 'relu':
        activation_func = relu
        deriv_activation_func = derivative_of_relu
    elif name == 'sigmoid':
        activation_func = sigmoid
        deriv_activation_func = derivative_of_sigmoid
    elif name == 'softmax':
        activation_func = softmax
        deriv_activation_func = None
    elif name == 'tanh':
        activation_func = tahn
        deriv_activation_func = derivative_of_tanh
    else:
        raise Exception("No Activation Function Given!")

    return (activation_func, deriv_activation_func)
