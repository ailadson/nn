from layer import FullConnectedLayer
from scipy.signal import convolve2d as c2d
import numpy as np
from functions import relu


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
        self.activation_func = relu
        self.total_input = np.zeros([num_of_output_layers, prev_shape[1], prev_shape[2]])

    def initialize_weights(self):
        return np.random.uniform(-1, 1, [
            self.output.shape[0],
            self.prev_layer.output.shape[0],
            self.height,
            self.width
        ])

    def forward_propagate(self):
        self.total_input *= 0
        for output_layer_idx in range(self.weights.shape[0]):
            for input_layer_idx in range(self.prev_layer.output.shape[0]):
                input_layer = self.prev_layer.output[input_layer_idx]
                kernel_weights = self.weights[output_layer_idx, input_layer_idx]
                self.total_input[output_layer_idx] += c2d(input_layer, kernel_weights, mode="same")
        self.activation_func(self.total_input, self.output)

    def back_propagate(self):
        deriv_filter = np.zeros(self.weights.shape)
        input_layers = self.prev_layer.output
        next_deriv_wrt_unit_total_inputs = self.next_layer.deriv_wrt_unit_total_inputs()
        for output_layer_idx in range(self.weights.shape[0]):
            for input_layer_idx in range(input_layers.shape[0]):
                input_layer = input_layers[input_layer_idx]
                bp_errors = next_deriv_wrt_unit_total_inputs[output_layer_idx]
                deriv_filter[output_layer_idx, input_layer_idx] += self.deriv_wrt_weights(input_layer, bp_errors)

    # def deriv_wrt_weights(self, input_layer, bp_errors):
    #     for i in range(self.height):
    #         for j in range(self.weight):
