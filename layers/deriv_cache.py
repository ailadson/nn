import config
from functions import zeros
import numpy as np

class DerivativeCache:
    def __init__(self, layer):
        self.prev_outputs = np.zeros(layer.prev_output_shape, dtype=config.FLOAT_TYPE)
        self.z_output = np.zeros(layer.output_shape, dtype=config.FLOAT_TYPE)
        if layer.has_weights():
            self.biases = np.zeros(layer.biases.shape, dtype=config.FLOAT_TYPE)
            self.weights = np.zeros(layer.weights.shape, dtype=config.FLOAT_TYPE)
        self.reset()

    def is_set(self, name):
        return self.is_set_d[name]

    def reset(self):
        self.is_set_d = {
            "weights" : False,
            "prev_outputs" : False,
            "z_output" : False,
            "biases" : False
        }

    def set(self, name):
        self.is_set_d[name] = True