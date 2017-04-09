import config
from functions import *
from layers.derivative_cache import DerivativeCache
from layers.layer import Layer
import numpy as np
import pyx.avx_convolve2d
import pyx.deconvolve2d

class ConvolutionLayer(Layer):
    def __init__(self,
                 prev_layer,
                 num_output_layers,
                 kernel_height,
                 kernel_width,
                 activation_func_name):
        output_shape = (
            num_output_layers,
            prev_layer.output_shape[1],
            prev_layer.output_shape[2]
        )

        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.num_input_layers = prev_layer.output_shape[0]
        self.num_output_layers = num_output_layers

        self.biases = config.float_zeros(self.num_output_layers)
        self.weights = initialize_weights(
            self.num_output_layers,
            self.num_input_layers,
            self.kernel_height,
            self.kernel_width
        )

        super().__init__(prev_layer, output_shape, activation_func_name)

    # Activation Functions
    def calculate_z_outputs(self, z_outputs):
        pyx.avx_convolve2d.apply_convolution(
            self.prev_layer.outputs(), self.weights, z_outputs
        )
        z_outputs += self.biases[:, None, None]

    def calculate_outputs(self, outputs):
        self.activation_func(self.z_outputs(), outputs)

    # Derivative Functions
    def calculate_deriv_wrt_weights(self, deriv_wrt_weights):
        pyx.deconvolve2d.deriv_wrt_weights(
            self.prev_layer.outputs(),
            deriv_wrt_weights,
            self.deriv_wrt_z_outputs()
        )

    def calculate_deriv_wrt_z_outputs(self, deriv_wrt_z_outputs):
        self.deriv_activation_func(
            self.z_outputs(), deriv_wrt_z_outputs
        )
        deriv_wrt_z_outputs *= self.deriv_wrt_outputs()

    def calculate_deriv_wrt_prev_outputs(self, deriv_wrt_prev_outputs):
        pyx.avx_convolve2d.apply_backward_convolution(
            self.deriv_wrt_z_outputs(),
            self.weights,
            deriv_wrt_prev_outputs
        )

    def calculate_deriv_wrt_biases(self, deriv_wrt_biases):
        # TODO: must implement this!
        deriv_wrt_biases.fill(0.0)

    # Others
    def has_weights(self):
        return True

# Helpers
def initialize_weights(
    num_output_layers, num_input_layers, kernel_height, kernel_width):

    return np.random.uniform(-1, 1, [
        num_output_layers,
        num_input_layers,
        kernel_height,
        kernel_width
    ]).astype(config.FLOAT_TYPE)
