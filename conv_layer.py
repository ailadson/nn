from scipy.signal import convolve2d as c2d
import numpy as np
from functions import *
from deriv_cache import ConvDerivativeCache


class ConvolutionalLayer():
    def __init__(self, prev_layer, height, width, num_of_output_layers):
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        self.height = height
        self.width = width
        prev_shape = prev_layer.output.shape
        self.num_of_input_layers = prev_shape[0]
        self.output = np.zeros([num_of_output_layers, prev_shape[1], prev_shape[2]])
        self.weights = self.initialize_weights()
        self.activation_func = lambda val, des: np.copyto(des, val)
        self.deriv_activation_func = lambda val, des: des.fill(1)
        # self.activation_func = relu
        # self.deriv_activation_func = deriv_of_relu
        self.total_input = np.zeros([num_of_output_layers, prev_shape[1], prev_shape[2]])
        self.deriv_cache = ConvDerivativeCache(self)

    def initialize_weights(self):
        return np.random.uniform(0, 1, [
            self.output.shape[0],
            self.prev_layer.output.shape[0],
            self.height,
            self.width
        ])

    def forward_propagate(self):
        self.total_input *= 0
        self.apply_convolution(self.prev_layer.output, self.total_input)
        self.activation_func(self.total_input, self.output)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_weights()
        # print("Print Derivative Weights")
        # print(self.deriv_cache.weights)

    def deriv_wrt_weights(self):
        if self.deriv_cache.is_set("weights"):
            return self.deriv_cache.weights

        input_layers = self.prev_layer.output
        deriv_wrt_unit_total_inputs = self.deriv_wrt_unit_total_inputs()
        self.deriv_cache.weights *= 0
        for output_layer_idx in range(self.weights.shape[0]):
            for input_layer_idx in range(input_layers.shape[0]):
                input_layer = input_layers[input_layer_idx]
                bp_errors = deriv_wrt_unit_total_inputs[output_layer_idx]
                deriv_filter = self.deriv_cache.weights[output_layer_idx, input_layer_idx]
                deconvolve2d(input_layer, bp_errors, deriv_filter)

    def deriv_wrt_unit_total_inputs(self):
        if self.deriv_cache.is_set("unit_total_inputs"):
            return self.deriv_cache.unit_total_inputs

        self.deriv_activation_func(self.total_input, self.deriv_cache.unit_total_inputs)
        self.deriv_cache.unit_total_inputs *= self.deriv_wrt_unit_outputs()

        return self.deriv_cache.unit_total_inputs

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs
        apply_convolution(self.deriv_wrt_unit_total_inputs(), self.deriv_cache.prev_outputs, True)

    def apply_convolution(self, ipt, des, flip = False):
        for output_layer_idx in range(self.weights.shape[0]):
            for input_layer_idx in range(ipt.shape[0]):
                input_layer = ipt[input_layer_idx]
                kernel_weights = self.weights[output_layer_idx, input_layer_idx]
                if flip:
                    kernel_weights = np.fliplr(np.flipud(kernel_weights))
                    kernel_weights = pad_flipped_weights(kernel_weights)
                des[output_layer_idx] += c2d(input_layer, kernel_weights, mode="same")

    def pad_flipped_weights(weights):
        height, width = weights.shape
        new_height = height + (1 if height % 2 == 0 else 0)
        new_width = width + (1 if width % 2 == 0 else 0)
        if height % 2 == 0:
            weights = np.append(weights, np.zeros([1, width]), axis=0)
        if width % 2 == 0:
            weights = np.append(weights, np.zeros([new_height, 1]), axis=1)
        return weights
