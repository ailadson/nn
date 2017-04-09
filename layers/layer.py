import config
from functions.activations import *
from layers.activation_cache import ActivationCache
from layers.derivative_cache import DerivativeCache

class Layer:
    def __init__(self, prev_layer, output_shape, activation_func_name):
        if prev_layer:
            self.prev_output_shape = prev_layer.output_shape
        else:
            self.prev_output_shape = None
        self.output_shape = output_shape

        activation_funcs = get_activation_functions(
            activation_func_name
        )
        self.activation_func = activation_funcs[0]
        self.deriv_activation_func = activation_funcs[1]

        self.prev_layer = prev_layer
        self.next_layer = None
        if prev_layer is not None:
            prev_layer.next_layer = self

        self.activation_cache = ActivationCache(self)
        self.derivative_cache = DerivativeCache(self)

    # Activation Functions
    def z_outputs(self):
        if self.activation_cache.is_set("z_outputs"):
            return self.activation_cache.z_outputs

        self.calculate_z_outputs(self.activation_cache.z_outputs)
        self.activation_cache.set("z_outputs")
        return self.activation_cache.z_outputs

    def calculate_z_outputs(self, z_outputs):
        raise Exception(f"Not Implemented for {self}!")

    def outputs(self):
        if self.activation_cache.is_set("outputs"):
            return self.activation_cache.outputs

        self.calculate_outputs(self.activation_cache.outputs)
        self.activation_cache.set("outputs")
        return self.activation_cache.outputs

    def calculate_outputs(self, outputs):
        raise Exception(f"Not Implemented for {self}!")

    # Derivative Functions
    def deriv_wrt_biases(self):
        if self.derivative_cache.is_set("biases"):
            return self.derivative_cache.biases

        self.calculate_deriv_wrt_biases(self.derivative_cache.biases)
        self.derivative_cache.set('biases')
        return self.derivative_cache.biases

    def calculate_deriv_wrt_biases(self, deriv_wrt_biases):
        raise Exception(f"Not Implemented for {self}!")

    def deriv_wrt_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_prev_outputs(self):
        if self.derivative_cache.is_set("prev_outputs"):
            return self.derivative_cache.prev_outputs

        self.calculate_deriv_wrt_prev_outputs(
            self.derivative_cache.prev_outputs
        )
        self.derivative_cache.set("prev_outputs")
        return self.derivative_cache.prev_outputs

    def calculate_deriv_wrt_prev_outputs(self, deriv_wrt_prev_outputs):
        raise Exception(f"Not Implemented for {self}!")

    def deriv_wrt_z_outputs(self):
        if self.derivative_cache.is_set("z_outputs"):
            return self.derivative_cache.z_outputs

        self.calculate_deriv_wrt_z_outputs(
            self.derivative_cache.z_outputs
        )
        self.derivative_cache.set("z_outputs")
        return self.derivative_cache.z_outputs

    def calculate_deriv_wrt_z_outputs(self, deriv_wrt_z_outputs):
        raise Exception(f"Not Implemented for {self}!")

    def deriv_wrt_weights(self):
        if self.derivative_cache.is_set("weights"):
            return self.derivative_cache.weights

        self.calculate_deriv_wrt_weights(self.derivative_cache.weights)
        self.derivative_cache.set("weights")
        return self.derivative_cache.weights

    def calculate_deriv_wrt_weights(self, deriv_wrt_weights):
        raise Exception(f"Not Implemented for {self}!")

    # Other
    def has_weights(self):
        raise Exception(f"Not Implemented for {self}!")

    def reset_caches(self):
        self.activation_cache.reset()
        self.derivative_cache.reset()

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
    elif name == None:
        activation_func = None
        deriv_activation_func = None
    else:
        raise Exception("No Activation Function Given!")

    return (activation_func, deriv_activation_func)
