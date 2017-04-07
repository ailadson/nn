from random import uniform
from functions import *
import math
from itertools import product
from deriv_cache import DerivativeCache
import numpy as np

class FullyConnectedLayer:
    @staticmethod
    def w_bound(num_units, prev_units):
        return 4 * math.sqrt(6 / (num_units + prev_units))

    def __init__(self, prev_layer, num_units):
        self.activation_func = relu
        self.deriv_activation_func = deriv_of_relu
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        self.num_units = num_units
        self.biases = np.zeros(num_units, dtype=np.float32)
        #weight col correspond to prev units. rows correspond to our units
        self.weights = self.generate_weight_mat()
        self.output = np.zeros([num_units], dtype=np.float32)
        self.total_input = np.zeros([num_units], dtype=np.float32)
        self.deriv_cache = DerivativeCache(self)


    def generate_weight_mat(self):
        self.w_bound = FullyConnectedLayer.w_bound(self.num_units, self.prev_layer.num_units)
        num_prev_units = self.prev_layer.num_units
        return np.random.uniform(-self.w_bound, self.w_bound, [self.num_units, num_prev_units]).astype(np.float32)

    def forward_propagate(self):
        self.weights.dot(self.prev_layer.output, out = self.total_input)
        self.total_input += self.biases
        self.activation_func(self.total_input, self.output)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_weights()
        self.deriv_wrt_biases()

    def deriv_wrt_biases(self):
        if self.deriv_cache.is_set("biases"):
            return self.deriv_cache.biases

        np.copyto(self.deriv_cache.biases, self.deriv_wrt_unit_total_inputs())
        return self.deriv_cache.biases

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs

        deriv_wrt_unit_total_inputs = self.deriv_wrt_unit_total_inputs()
        t_weights = self.weights.transpose()
        t_weights.dot(deriv_wrt_unit_total_inputs, out = self.deriv_cache.prev_outputs)
        self.deriv_cache.set('prev_outputs')
        return self.deriv_cache.prev_outputs

    def deriv_wrt_unit_total_inputs(self):
        if self.deriv_cache.is_set("unit_total_inputs"):
            return self.deriv_cache.unit_total_inputs

        self.deriv_activation_func(self.total_input, self.deriv_cache.unit_total_inputs)
        self.deriv_cache.unit_total_inputs *= self.deriv_wrt_unit_outputs()

        return self.deriv_cache.unit_total_inputs

    def deriv_wrt_weights(self):
        if self.deriv_cache.is_set("weights"):
            return self.deriv_cache.weights

        deriv_wrt_unit_inputs = self.deriv_wrt_unit_total_inputs()
        np.outer(deriv_wrt_unit_inputs, self.prev_layer.output, self.deriv_cache.weights)
        return self.deriv_cache.weights

    def has_weights(self):
        return True
