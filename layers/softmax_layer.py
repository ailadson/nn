import config
from layers.layer import Layer
from functions.activations import *
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer, [prev_layer.output_shape[0]], 'softmax')
        self.observed_output = None
        self.true_class = None

    def back_propagate(self):
        pass

    def forward_propagate(self):
        np.copyto(self.z_output, self.prev_layer.output)
        # to prevent e^(z_output) from being inf, we shift z_output which
        # wont change softmax result
        self.z_output -= max(self.z_output)
        self.activation_func(self.z_output, self.output)

    def set_observed_output(self, observed):
        self.observed_output = observed
        self.true_class = list(observed).index(1)

    def deriv_wrt_prev_outputs(self):
        unit_outputs = np.zeros(self.output_shape, dtype=config.FLOAT_TYPE)
        derivative_of_softmax_and_ce(self.z_output, self.true_class, unit_outputs)
        return unit_outputs

    def has_weights(self):
        return False
