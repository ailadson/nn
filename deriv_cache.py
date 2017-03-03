from functions import zeros
import numpy as np

class DerivativeCache:
    def __init__(self, layer):
        self.weights = np.zeros([layer.num_units, layer.prev_layer.num_units])
        self.unit_outputs = np.zeros([layer.num_units])
        self.unit_total_inputs = np.zeros([layer.num_units])
        self.reset()

    def reset(self):
        self.is_set_d = { "weights" : False,
                        "unit_outputs" : False,
                        "unit_total_inputs" : False
                    }

    def is_set(self, name):
        return self.is_set_d[name]

    def __str__(self):
        return f"Weights: {self.weights}\nUnit_outputs: {self.unit_outputs}\nunit_total_inputs: {self.unit_total_inputs}"
