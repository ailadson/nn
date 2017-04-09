import config
import numpy as np

class DerivativeCache:
    def __init__(self, layer):
        self.layer = layer

        self.outputs = np.zeros(
            layer.output_shape, dtype=config.FLOAT_TYPE
        )
        self.prev_outputs = np.zeros(
            layer.prev_output_shape, dtype=config.FLOAT_TYPE
        )
        self.z_outputs = np.zeros(
            layer.output_shape, dtype=config.FLOAT_TYPE
        )

        if layer.has_weights():
            self.biases = np.zeros(
                layer.biases.shape, dtype=config.FLOAT_TYPE
            )
            self.weights = np.zeros(
                layer.weights.shape, dtype=config.FLOAT_TYPE
            )

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
