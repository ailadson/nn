from deriv_cache import MaxPoolDerivativeCache
import math
import numpy as np
import pyx.max_pooling_functions

class MaxPoolingLayer():
    def __init__(self, prev_layer):
        output_shape = (
            prev_layer.output_shape[0],
            math.ceil(prev_layer.output_shape[1] / 2),
            math.ceil(prev_layer.output_shape[2] / 2)
        )

        super().Layer.__init__(prev_layer, output_shape, 'id')
        self.deriv_cache = MaxPoolDerivativeCache(self)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_prev_outputs()

    def forward_propagate(self):
        self.z_output *= 0;
        pyx.max_pooling_functions.apply_max_pooling(
            self.prev_layer.output, self.z_output
        )
        self.activation_func(self.z_output, self.output)

    # Derivative Functions
    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs

        self.deriv_cache.prev_outputs *= 0
        pyx.max_pooling_functions.back_propagate_channels(
            self.deriv_cache.prev_outputs,
            self.prev_layer.output,
            self.deriv_wrt_unit_outputs()
        )

        self.deriv_cache.set('prev_outputs')
        return self.deriv_cache.prev_outputs

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def has_weights(self):
        return False
