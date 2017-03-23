from functions import zeros
import numpy as np

class DerivativeCache:
    def __init__(self, layer):
        self.weights = np.zeros([layer.num_units, layer.prev_layer.num_units])
        self.prev_outputs = np.zeros([layer.prev_layer.num_units])
        self.unit_total_inputs = np.zeros([layer.num_units])
        self.reset()

    def reset(self):
        self.is_set_d = { "weights" : False,
                        "prev_outputs" : False,
                        "unit_total_inputs" : False
                    }

    def is_set(self, name):
        return self.is_set_d[name]

    def __str__(self):
        return f"Weights: {self.weights}\nPrev_outputs: {self.prev_outputs}\nunit_total_inputs: {self.unit_total_inputs}"

class ConvDerivativeCache(DerivativeCache):
    def __init__(self, layer):
        self.weights = np.zeros(layer.weights.shape)
        self.prev_outputs = np.zeros(layer.output.shape)
        self.unit_total_inputs = np.zeros(layer.total_input.shape)
        self.reset()
