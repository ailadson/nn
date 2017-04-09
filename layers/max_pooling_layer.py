from layers.derivative_cache import DerivativeCache
from layers.layer import Layer
import math
import numpy as np
import pyx.max_pooling_functions

class MaxPoolingLayer(Layer):
    def __init__(self, prev_layer):
        output_shape = (
            prev_layer.output_shape[0],
            math.ceil(prev_layer.output_shape[1] / 2),
            math.ceil(prev_layer.output_shape[2] / 2)
        )

        super().__init__(prev_layer, output_shape, 'id')

    # Activation Functions
    def calculate_z_outputs(self, z_outputs):
        pyx.max_pooling_functions.apply_max_pooling(
            self.prev_layer.outputs(), z_outputs
        )

    def calculate_outputs(self, outputs):
        self.activation_func(self.z_outputs(), self.outputs())

    # Derivative Functions
    def calculate_deriv_wrt_prev_outputs(self, deriv_wrt_prev_outputs):
        pyx.max_pooling_functions.back_propagate_channels(
            deriv_wrt_prev_outputs,
            self.prev_layer.outputs(),
            self.deriv_wrt_unit_outputs()
        )

    # Other
    def has_weights(self):
        return False
