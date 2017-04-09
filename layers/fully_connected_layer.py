import config
from functions import *
from layers.layer import Layer
import math
import numpy as np

class FullyConnectedLayer(Layer):
    def __init__(self, prev_layer, num_units, activation_func_name):
        super().__init__(prev_layer, [num_units], activation_func_name)
        self.biases = np.zeros([num_units], dtype=config.FLOAT_TYPE)
        self.weights = generate_weight_matrix(
            num_units, prev_layer.output_shape
        )

    # Activation Functions
    def calculate_z_outputs(self, z_outputs):
        self.weights.dot(self.prev_layer.output, out = z_outputs)
        z_outputs += self.biases

    def calculate_outputs(self, outputs):
        self.activation_func(self.z_output, self.outputs)

    # Derivative Functions
    def calculate_deriv_wrt_biases(self, deriv_wrt_biases):
        np.copyto(self.deriv_cache.biases, self.deriv_wrt_z_outputs())

    def calculate_deriv_wrt_prev_outputs(self, deriv_wrt_prev_outputs):
        deriv_wrt_z_outputs = self.deriv_wrt_z_outputs()
        t_weights = self.weights.transpose()
        t_weights.dot(
            deriv_wrt_z_outputs,
            out = deriv_wrt_prev_outputs
        )

    def calculate_deriv_wrt_weights(self, deriv_wrt_weights):
        deriv_wrt_unit_inputs = self.deriv_wrt_z_outputs()
        np.outer(
            deriv_wrt_unit_inputs,
            self.prev_layer.output,
            deriv_wrt_weights
        )

    def calculate_deriv_wrt_z_outputs(self, deriv_wrt_z_outputs):
        self.deriv_activation_func(
            self.z_output, deriv_wrt_z_outputs
        )
        self.deriv_cache.z_outputs *= self.deriv_wrt_unit_outputs()

    # Other
    def has_weights(self):
        return True

# Helpers
def generate_weight_matrix(num_units, output_shape):
    assert(len(output_shape) == 1)
    prev_num_units = output_shape[0]

    w_bound = weight_bound(
        num_units, prev_num_units
    )
    return np.random.uniform(
        -w_bound, w_bound, [num_units, prev_num_units]
    ).astype(config.FLOAT_TYPE)

def weight_bound(num_units, prev_num_units):
    return 4 * math.sqrt(6 / (num_units + prev_num_units))
