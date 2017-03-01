from random import uniform
from functions import *
import math
from itertools import product
from deriv_cache import DerivativeCache

class Layer:
    @staticmethod
    def w_bound(num_units, prev_units):
        return 4 * math.sqrt(6 / (num_units + prev_units))

    def __init__(self, prev_layer, num_units):
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        self.num_units = num_units
        self.w_bound = Layer.w_bound(num_units, prev_layer.num_units)
        #weight col correspond to prev units. rows correspond to our units
        self.weights = [self.generate_weight_row() for i in range(num_units)]
        self.output = zeros([num_units])
        self.total_input = zeros([num_units])
        self.deriv_cache = DerivativeCache(self)


    def generate_weight_row(self):
        num_prev_units = self.prev_layer.num_units
        return [uniform(-self.w_bound, self.w_bound) for j in range(num_prev_units)]

    def forward_propagate(self):
        matrix_times_vec(self.weights, self.prev_layer.output, self.input)
        sigmoid(self.input, self.output)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_weights()

    def deriv_wrt_unit_outputs(self):
        if self.deriv_cache.is_set("unit_outputs"):
            return self.deriv_cache.unit_outputs

        next_deriv_wrt_unit_total_inputs = self.next_layer.deriv_wrt_unit_total_inputs

        for i in range(self.num_units):
            sum_v = 0
            for j in range(self.next_layer.num_units):
                sum_v += (next_deriv_wrt_unit_total_inputs[j] * self.weights[i][j])
            self.deriv_cache.unit_outputs[i] =  sum_v

        return self.deriv_cache.unit_outputs

    def deriv_wrt_unit_total_inputs(self):
        if self.deriv_cache.is_set("unit_total_inputs"):
            return self.deriv_cache.unit_total_inputs

        deriv_wrt_unit_outputs = self.deriv_wrt_unit_outputs()

        for i, ipt in enumerate(self.total_input):
            deriv_wrt_unit_input = derivative_of_sig(ipt) * deriv_wrt_unit_output[i]
            self.deriv_cache.unit_total_inputs[i] = deriv_wrt_unit_input

        return self.deriv_cache.unit_total_inputs

    def deriv_wrt_weights(self):
        if self.deriv_cache.is_set("weights"):
            return self.deriv_cache.weights

        deriv_wrt_unit_inputs = self.deriv_wrt_unit_inputs()

        for i in range(self.num_units):
            for j in range(self.prev_layer.num_units):
                 self.deriv_cache.weights[i][j] = deriv_wrt_unit_inputs[i] * self.prev_layer.output[j]

        return self.deriv_cache.weights
