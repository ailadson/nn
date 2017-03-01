from Layer import Layer
from functions import *

class OutputLayer(Layer):
    def __init__(self, prev_layer, num_units):
        self.super(prev_layer, num_units)
        self.observed_output = None


    # def back_propagation(self, observed):
    #     loss_deriv = derivative_of_ce(self.output[0], observed)

    def set_observed_output(self, observed):
        self.observed_output = observed

    def deriv_wrt_unit_outputs(self):
        return [derivative_of_ce(self.output[0], self.observed_output)]

    def deriv_wrt_unit_total_inputs(self):
        deriv_wrt_unit_inputs = []
        deriv_wrt_unit_outputs = self.deriv_wrt_unit_outputs()

        for i, ipt in enumerate(self.total_input):
            deriv_wrt_unit_input = derivative_of_sig(ipt) * deriv_wrt_unit_output[i]
            deriv_wrt_unit_inputs.append(deriv_wrt_unit_input)

        return deriv_wrt_unit_inputs

    def deriv_wrt_weights(self):
        num_rows = self.num_units
        num_cols = self.prev_layer.num_units
        deriv_wrt_weights_mat = [[0] * num_cols for i in range(num_rows)]
        deriv_wrt_unit_inputs = self.deriv_wrt_unit_inputs()

        for i in range(num_rows):
            for j in range(num_cols):
                 deriv_wrt_weights_mat[i][j] = deriv_wrt_unit_inputs[i] * self.prev_layer.output[j]

        return deriv_wrt_weights_mat
