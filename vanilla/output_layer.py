from layer import Layer
from functions import *

class OutputLayer(Layer):
    def __init__(self, prev_layer, num_units):
        super().__init__(prev_layer, num_units)
        self.observed_output = None

    def set_observed_output(self, observed):
        self.observed_output = observed

    def deriv_wrt_unit_outputs(self):
        if self.deriv_cache.is_set("unit_outputs") is False:
            self.deriv_cache.unit_outputs[0] = derivative_of_ce(self.output[0], self.observed_output)
        return self.deriv_cache.unit_outputs
