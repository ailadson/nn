import config
from layers.layer import Layer
from functions.activations import *
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self, prev_layer):
        num_classes = prev_layer.output_shape[0]
        self.observed_output = None
        self.true_class = None

        super().__init__(prev_layer, [num_classes], 'softmax')

    # Activation Functions
    def calculate_z_outputs(self, z_outputs):
        np.copyto(z_outputs, self.prev_layer.outputs())
        # to prevent e^(z_output) from being inf, we shift z_output
        # which won't change eventual softmax result.
        z_outputs -= max(z_outputs)

    def calculate_outputs(self, outputs):
        self.activation_func(self.z_outputs(), outputs)

    def logits(self):
        return self.z_outputs()

    # Derivative Functions
    def calculate_deriv_wrt_outputs(self, deriv_wrt_outputs):
        # this can be very inaccurate because it converts from
        # logscale to probabilities. avoid using this!
        derivative_of_ce(
            self.outputs(),
            self.observed_output,
            out = deriv_wrt_outputs
        )

    def deriv_wrt_prev_outputs(self):
        # this is the same because no weights here.
        return self.deriv_wrt_z_outputs()

    def calculate_deriv_wrt_z_outputs(self, deriv_wrt_z_outputs):
        derivative_of_softmax_and_ce(
            self.z_outputs(), self.true_class, deriv_wrt_z_outputs
        )

    # Other
    def has_weights(self):
        return False

    def set_observed_output(self, observed):
        self.observed_output = observed
        self.true_class = list(observed).index(1)
