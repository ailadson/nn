from random import uniform
from functions import *
import math
from itertools import product
from deriv_cache import DerivativeCache
import numpy as np

class FullConnectedLayer:
    @staticmethod
    def w_bound(num_units, prev_units):
        return 4 * math.sqrt(6 / (num_units + prev_units))

    def __init__(self, prev_layer, num_units):
        self.activation_func = sigmoid
        self.deriv_activation_func = derivative_of_sig
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        self.num_units = num_units
        self.biases = np.zeros(num_units)
        #weight col correspond to prev units. rows correspond to our units
        self.weights = self.generate_weight_mat()
        self.output = np.zeros([num_units])
        self.total_input = np.zeros([num_units])
        self.deriv_cache = DerivativeCache(self)


    def generate_weight_mat(self):
        self.w_bound = Layer.w_bound(num_units, prev_layer.num_units)
        num_prev_units = self.prev_layer.num_units
        return np.random.uniform(-self.w_bound, self.w_bound, [self.num_units, num_prev_units])

    def forward_propagate(self):
        self.weights.dot(self.prev_layer.output, out = self.total_input)
        self.total_input += self.biases
        self.activation_func(self.total_input, self.output)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_weights()

    def deriv_wrt_unit_outputs(self):
        if self.deriv_cache.is_set("unit_outputs"):
            return self.deriv_cache.unit_outputs

        next_deriv_wrt_unit_total_inputs = self.next_layer.deriv_wrt_unit_total_inputs()
        t_weights = self.next_layer.weights.transpose()
        t_weights.dot(next_deriv_wrt_unit_total_inputs, out = self.deriv_cache.unit_outputs)

        return self.deriv_cache.unit_outputs

    def deriv_wrt_unit_total_inputs(self):
        if self.deriv_cache.is_set("unit_total_inputs"):
            return self.deriv_cache.unit_total_inputs

        deriv_wrt_unit_outputs = self.deriv_wrt_unit_outputs()
        self.deriv_activation_func(self.total_input, self.deriv_cache.unit_total_inputs)
        self.deriv_cache.unit_total_inputs *= deriv_wrt_unit_outputs

        return self.deriv_cache.unit_total_inputs

    def deriv_wrt_weights(self):
        if self.deriv_cache.is_set("weights"):
            return self.deriv_cache.weights

        deriv_wrt_unit_inputs = self.deriv_wrt_unit_total_inputs()
        np.outer(deriv_wrt_unit_inputs, self.prev_layer.output, self.deriv_cache.weights)
        return self.deriv_cache.weights
