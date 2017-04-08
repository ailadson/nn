from deriv_cache import DerivativeCache
from functions import *
from itertools import product
from layer import Layer
import math
import numpy as np
from random import uniform

class FullyConnectedLayer(Layer):
    def __init__(self, prev_layer, num_units, activation_func_name):
        super.Layer.__init__(prev_layer, num_units, activation_func_name)
        self.biases = np.zeros(num_units, dtype=np.float32)
        self.weights = self.generate_weight_mat()
        self.deriv_cache = DerivativeCache(self)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_weights()
        self.deriv_wrt_biases()

    def forward_propagate(self):
        self.weights.dot(self.prev_layer.output, out = self.total_input)
        self.total_input += self.biases
        self.activation_func(self.total_input, self.output)

    # Derivative Functions
    def deriv_wrt_biases(self):
        if self.deriv_cache.is_set("biases"):
            return self.deriv_cache.biases

        np.copyto(self.deriv_cache.biases, self.deriv_wrt_unit_total_inputs())

        self.deriv_cache.set('biases')
        return self.deriv_cache.biases

    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs

        deriv_wrt_unit_total_inputs = self.deriv_wrt_unit_total_inputs()
        t_weights = self.weights.transpose()
        t_weights.dot(
            deriv_wrt_unit_total_inputs,
            out = self.deriv_cache.prev_outputs
        )

        self.deriv_cache.set('prev_outputs')
        return self.deriv_cache.prev_outputs

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_unit_total_inputs(self):
        if self.deriv_cache.is_set("unit_total_inputs"):
            return self.deriv_cache.unit_total_inputs

        self.deriv_activation_func(
            self.total_input, self.deriv_cache.unit_total_inputs
        )
        self.deriv_cache.unit_total_inputs *= self.deriv_wrt_unit_outputs()

        self.deriv_cache.set("unit_total_inputs")
        return self.deriv_cache.unit_total_inputs

    def deriv_wrt_weights(self):
        if self.deriv_cache.is_set("weights"):
            return self.deriv_cache.weights

        deriv_wrt_unit_inputs = self.deriv_wrt_unit_total_inputs()
        np.outer(
            deriv_wrt_unit_inputs,
            self.prev_layer.output,
            self.deriv_cache.weights
        )

        self.deriv_cache.set("weights")
        return self.deriv_cache.weights

    def has_weights(self):
        return True

def generate_weight_mat(num_units, prev_num_units):
    w_bound = weight_bound(
        num_units, prev_num_units
    )
    return np.random.uniform(
        -w_bound, w_bound, [num_units, prev_num_units]
    ).astype(np.float32)

def weight_bound(num_units, prev_num_units):
    return 4 * math.sqrt(6 / (num_units + prev_num_units))
