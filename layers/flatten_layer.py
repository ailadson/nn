from functools import reduce
from layers.layer import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self, prev_layer):
        num_units = reduce(lambda acc, ele: acc * ele, prev_layer.output_shape)
        super().__init__(prev_layer, [num_units], 'id')

    # Activation Functions
    def calculate_outputs(self, outputs):
        outputs[:] = (
            self.prev_layer.output.reshape(self.output_shape)[:]
        )

    # Derivative Functions
    def calculate_deriv_wrt_prev_outputs(self, deriv_wrt_prev_outputs):
        deriv_wrt_prev_outputs[:] = (
            self.next_layer.deriv_wrt_prev_outputs()
        ).reshape(self.prev_layer.output.shape)[:]

    # Other
    def has_weights(self):
        return False
