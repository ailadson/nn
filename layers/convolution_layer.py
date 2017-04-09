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

        super().__init__(prev_layer, output_shape, activation_func_name)
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.num_input_layers = self.prev_output_shape[0]
        self.num_output_layers = num_output_layers

        self.biases = np.zeros(self.num_output_layers, dtype=config.FLOAT_TYPE)
        self.weights = initialize_weights(
            self.num_output_layers,
            self.num_input_layers,
            self.kernel_height,
            self.kernel_width
        )
        self.deriv_cache = DerivativeCache(self)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_weights()

    def forward_propagate(self):
        self.z_output *= 0
        pyx.avx_convolve2d.apply_convolution(
            self.prev_layer.output, self.weights, self.z_output
        )
        self.z_output += self.biases[:, None, None]
        self.activation_func(self.z_output, self.output)

    def deriv_wrt_weights(self):
        if self.deriv_cache.is_set("weights"):
            return self.deriv_cache.weights

        self.deriv_cache.weights.fill(0.0)
        # TODO: we don't do anything to update the biases!
        pyx.deconvolve2d.deriv_wrt_weights(
            self.prev_layer.output,
            self.deriv_cache.weights,
            self.deriv_wrt_z_outputs()
        )

        self.deriv_cache.set('weights')
        return self.deriv_cache.weights

    def deriv_wrt_z_outputs(self):
        if self.deriv_cache.is_set("z_outputs"):
            return self.deriv_cache.z_outputs

        self.deriv_activation_func(
            self.z_output, self.deriv_cache.z_outputs
        )
        self.deriv_cache.z_outputs *= (
            self.deriv_wrt_unit_outputs()
        )

        return self.deriv_cache.z_outputs

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs

        self.deriv_cache.prev_outputs *= 0
        pyx.avx_convolve2d.apply_backward_convolution(
            self.deriv_wrt_z_outputs(),
            self.weights,
            self.deriv_cache.prev_outputs
        )

        self.deriv_cache.set('prev_outputs')
        return self.deriv_cache.prev_outputs

    def has_weights(self):
        return True

def initialize_weights(
    num_output_layers,
    num_input_layers,
    kernel_height,
    kernel_width):
    return np.random.uniform(-1, 1, [
        num_output_layers,
        num_input_layers,
        kernel_height,
        kernel_width
    ]).astype(config.FLOAT_TYPE)
