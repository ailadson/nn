import config
from layers.derivative_cache import DerivativeCache
from layers.layer import Layer
from functions.activations import *
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer, [prev_layer.output_shape[0]], 'softmax')
        self.observed_output = None
        self.true_class = None
        self.deriv_cache = DerivativeCache(self)

    def back_propagate(self):
        self.deriv_cache.reset()

    def forward_propagate(self):
        np.copyto(self.z_output, self.prev_layer.output)
        # to prevent e^(z_output) from being inf, we shift z_output which
        # wont change softmax result
        self.z_output -= max(self.z_output)
        self.activation_func(self.z_output, self.output)

    def logits(self):
        return self.z_output

    def set_observed_output(self, observed):
        self.observed_output = observed
        self.true_class = list(observed).index(1)

    def deriv_wrt_outputs(self):
        # this can be very inaccurate because it converts from
        # logscale to probabilities. avoid using this!
        if self.deriv_cache.is_set("outputs"):
            return self.deriv_cache.outputs
        derivative_of_ce(
            self.output,
            self.observed_output,
            out = self.deriv_cache.outputs
        )

        self.deriv_cache.set("outputs")
        return self.deriv_cache.outputs

    def deriv_wrt_prev_outputs(self):
        # this is the same because no weights here.
        return self.deriv_wrt_z_outputs()

    def deriv_wrt_z_outputs(self):
        if self.deriv_cache.is_set("z_outputs"):
            return self.deriv_cache.z_outputs

        derivative_of_softmax_and_ce(
            self.z_output, self.true_class, self.deriv_cache.z_outputs
        )

        self.deriv_cache.set("z_outputs")
        return self.deriv_cache.z_outputs

    def has_weights(self):
        return False
