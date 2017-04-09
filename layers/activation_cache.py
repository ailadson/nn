import config
import numpy as np

class ActivationCache:
    def __init__(self, layer):
        self.z_outputs = config.float_zeros(layer.output_shape)
        self.outputs = config.float_zeros(layer.output_shape)
        self.reset()

    def is_set(self, name):
        return self.is_set_d[name]

    def reset(self):
        self.z_outputs.fill(0.0)
        self.outputs.fill(0.0)
        self.is_set_d = {
            "z_outputs": False,
            "outputs": False
        }

    def set(self, name):
        self.is_set_d[name] = True
