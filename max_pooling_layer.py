from deriv_cache import MaxPoolDerivativeCache
from functions import *
import math
import numpy as np
import pyx.max_pooling_functions
import skimage.measure

class MaxPoolingLayer():
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        prev_shape = prev_layer.output.shape
        self.num_of_input_layers = prev_shape[0]
        output_shape = [
            prev_shape[0],
            math.ceil(prev_shape[1] / 2),
            math.ceil(prev_shape[2] / 2)
        ]
        self.output = np.zeros(output_shape, dtype=np.float32)
        self.activation_func = lambda val, des: np.copyto(des, val)
        self.deriv_activation_func = lambda val, des: des.fill(1)
        self.total_input = np.zeros(self.output.shape, dtype=np.float32)
        self.deriv_cache = MaxPoolDerivativeCache(self)

    def forward_propagate(self):
        self.total_input *= 0;
        self.apply_pooling(self.prev_layer.output, self.total_input)
        self.activation_func(self.total_input, self.output)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_prev_outputs()
        # print("Print Derivative Weights")
        # print(self.deriv_cache.weights)

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

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

    def apply_pooling(self, ipt, des):
        pyx.max_pooling_functions.apply_max_pooling(ipt, des)

    def has_weights(self):
        return False
