import config
import numpy as np

class DerivativeCache:
    def __init__(self, layer):
        self.layer = layer

        self.outputs = config.float_zeros(layer.output_shape)
        self.prev_outputs = config.float_zeros(layer.prev_output_shape)
        self.z_outputs = config.float_zeros(layer.output_shape)

        if layer.has_weights():
            self.biases = config.float_zeros(layer.biases.shape)
            self.weights = config.float_zeros(layer.weights.shape)

        self.reset()

    def is_set(self, name):
        return self.is_set_d[name]

    def reset(self):
        self.outputs.fill(0.0)
        self.prev_outputs.fill(0.0)
        self.z_outputs.fill(0.0)
        if self.layer.has_weights():
            self.biases.fill(0.0)
            self.weights.fill(0.0)

        self.is_set_d = {
            "weights": False,
            "outputs": False,
            "prev_outputs": False,
            "z_outputs": False,
            "biases": False
        }

    def set(self, name):
        self.is_set_d[name] = True
