from layer import Layer
from functions import *

class OutputLayer(Layer):
    def __init__(self, prev_layer, num_units):
        super().__init__(prev_layer, num_units)
        self.observed_output = None
        self.true_class = None
        self.activation_func = sigmoid # lambda val, des: np.copyto(des, val)
        self.deriv_activation_func = derivative_of_sig # lambda val, des: des.fill(1)

    def forward_propagate(self):
        super().forward_propagate()
        self.output -= max(self.output)
        softmax(self.output, self.output)

    def set_observed_output(self, observed):
        self.observed_output = observed
        self.true_class = list(observed).index(1)

    def deriv_wrt_unit_outputs(self):
        if self.deriv_cache.is_set("unit_outputs") is False:
             derivative_of_softmax_and_ce(self.output, self.true_class, self.deriv_cache.unit_outputs)
        return self.deriv_cache.unit_outputs
