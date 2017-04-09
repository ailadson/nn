from functools import reduce
from layers.layer import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self, prev_layer):
        num_units = reduce(lambda acc, ele: acc * ele, prev_layer.output_shape)
        super().__init__(prev_layer, [num_units], 'id')

    def back_propagate(self):
        pass

    def forward_propagate(self):
        self.output[:] = self.prev_layer.output.reshape(self.output_shape)[:]

    def deriv_wrt_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_prev_outputs(self):
        deriv_wrt_prev_outputs = (
            self.next_layer.deriv_wrt_prev_outputs()
        ).reshape(self.prev_layer.output.shape)
        return deriv_wrt_prev_outputs

    def has_weights(self):
        return False
