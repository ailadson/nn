from functools import reduce
import numpy as np

class FlattenLayer(Layer):
    def __init__(self, prev_layer):
        num_units = reduce(lambda acc, ele: acc * ele, prev_layer.output_shape)
        super().Layer.__init(prev_layer, [num_units], 'id')

    def back_propagate(self):
        pass

    def forward_propagate(self):
        self.output[:] = self.prev_layer.output.reshape(self.output_shape)[:]

    def deriv_wrt_prev_outputs(self):
        next_deriv_wrt_unit_total_inputs = self.next_layer.deriv_wrt_prev_outputs()
        return next_deriv_wrt_unit_total_inputs.reshape(self.prev_layer.output.shape)

    def has_weights(self):
        return False
