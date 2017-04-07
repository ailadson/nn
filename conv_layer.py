import pyx.deconvolve2d
from deriv_cache import ConvDerivativeCache
from functions import *
import numpy as np
import pyx.avx_convolve2d

class ConvolutionalLayer():
    def __init__(self, prev_layer, height, width, num_of_output_layers):
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        self.height = height
        self.width = width
        prev_shape = prev_layer.output.shape
        self.num_of_input_layers = prev_shape[0]
        output_shape = [
            num_of_output_layers, prev_shape[1], prev_shape[2]
        ]
        self.output = np.zeros(output_shape, dtype=np.float32)
        self.weights = self.initialize_weights()
        self.biases = np.zeros(
            (self.weights.shape[0], 1, 1), dtype=np.float32
        )
        self.activation_func = lambda val, des: np.copyto(des, val)
        self.deriv_activation_func = lambda val, des: des.fill(1)
        # self.activation_func = relu
        # self.deriv_activation_func = deriv_of_relu
        total_input_shape = [
            num_of_output_layers, prev_shape[1], prev_shape[2]
        ]
        self.total_input = np.zeros(total_input_shape, dtype=np.float32)
        self.deriv_cache = ConvDerivativeCache(self)

    def initialize_weights(self):
        return np.random.uniform(0, 1, [
            self.output.shape[0],
            self.prev_layer.output.shape[0],
            self.height,
            self.width
        ]).astype(np.float32)

    def forward_propagate(self):
        self.total_input *= 0
        self.apply_convolution(self.prev_layer.output, self.total_input)
        self.total_input += self.biases
        self.activation_func(self.total_input, self.output)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_weights()
        # print("Print Derivative Weights")
        # print(self.deriv_cache.weights)

    def deriv_wrt_weights(self):
        if self.deriv_cache.is_set("weights"):
            return self.deriv_cache.weights

        self.deriv_cache.weights.fill(0.0)
        # TODO: we don't do anything to update the biases!
        pyx.deconvolve2d.deriv_wrt_weights(
            self.prev_layer.output,
            self.deriv_cache.weights,
            self.deriv_wrt_unit_total_inputs()
        )
        self.deriv_cache.set('weights')
        return self.deriv_cache.weights

    def deriv_wrt_unit_total_inputs(self):
        if self.deriv_cache.is_set("unit_total_inputs"):
            return self.deriv_cache.unit_total_inputs

        self.deriv_activation_func(
            self.total_input, self.deriv_cache.unit_total_inputs
        )
        self.deriv_cache.unit_total_inputs *= (
            self.deriv_wrt_unit_outputs()
        )

        return self.deriv_cache.unit_total_inputs

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs
        self.deriv_cache.prev_outputs *= 0
        self.apply_backward_convolution(
            self.deriv_wrt_unit_total_inputs(),
            self.deriv_cache.prev_outputs
        )
        self.deriv_cache.set('prev_outputs')
        return self.deriv_cache.prev_outputs

    def apply_backward_convolution(
            self,
            deriv_wrt_total_inputs,
            deriv_wrt_prev_outputs):

        pyx.avx_convolve2d.apply_backward_convolution(
            deriv_wrt_total_inputs,
            self.weights,
            deriv_wrt_prev_outputs
        )

    def apply_convolution(self, ipt, des):
        pyx.avx_convolve2d.apply_convolution(ipt, self.weights, des)

    def has_weights(self):
        return True
