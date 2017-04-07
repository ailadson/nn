from layer import FullyConnectedLayer
from functions import *

class OutputLayer(FullyConnectedLayer):
    def __init__(self, prev_layer, num_units):
        super().__init__(prev_layer, num_units)
        self.observed_output = None
        self.true_class = None
        self.logits = np.zeros(num_units, dtype=np.float32)
        self.activation_func = lambda val, des: np.copyto(des, val)
        self.deriv_activation_func = lambda val, des: des.fill(1)

    def forward_propagate(self):
        super().forward_propagate()
        # outputs is in logits. make greatest value 0
        self.output -= max(self.output)
        # before softmax, hold on to logits
        np.copyto(self.logits , self.output)
        softmax(self.output, self.output)

    def set_observed_output(self, observed):
        self.observed_output = observed
        self.true_class = list(observed).index(1)

    def deriv_wrt_unit_outputs(self):
        unit_outputs = np.zeros([self.num_units], dtype=np.float32)
        derivative_of_softmax_and_ce(self.logits, self.true_class, unit_outputs)
        return unit_outputs
