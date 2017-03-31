from functools import reduce
import numpy as np

class FlattenLayer():
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        self.num_units = reduce(lambda acc, ele: acc * ele, self.prev_layer.output.shape)
        self.output = np.zeros(self.num_units)

    def forward_propagate(self):
        self.output[:] = self.prev_layer.output.reshape(self.num_units)[:]

    def back_propagate(self):
        pass

    def deriv_wrt_prev_outputs(self):
        next_deriv_wrt_unit_total_inputs = self.next_layer.deriv_wrt_prev_outputs()
        return next_deriv_wrt_unit_total_inputs.reshape(self.prev_layer.output.shape)

    def has_weights(self):
        return False
