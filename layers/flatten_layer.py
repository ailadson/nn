from functools import reduce
from layers.layer import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self, prev_layer):
        num_units = 1
        for dim in prev_layer.output_shape: num_units *= dim

        super().__init__(prev_layer, [num_units], 'id')

    # Activation Functions
    def calculate_outputs(self, outputs):
        outputs[:] = (
            self.prev_layer.outputs().reshape(self.output_shape)[:]
        )

    # Derivative Functions
    def calculate_deriv_wrt_prev_outputs(self, deriv_wrt_prev_outputs):
        deriv_wrt_prev_outputs[:] = (
            self.next_layer.deriv_wrt_prev_outputs()
        ).reshape(self.prev_layer.output_shape)[:]

    # Other
    def has_weights(self):
        return False
