import config
import numpy as np

class ActivationCache:
    def __init__(self, layer):
        self.z_output = np.zeros(
            layer.output_shape, dtype=config.FLOAT_TYPE
        )
        self.output = np.zeros(
            layer.output_shape, dtype=config.FLOAT_TYPE
        )

        self.reset()

    def is_set(self, name):
        return self.is_set_d[name]

    def reset(self):
        self.z_output.fill(0.0)
        self.output.fill(0.0)
        self.is_set_d = {
            "z_outputs": False,
            "outputs": False
        }

    def set(self, name):
        self.is_set_d[name] = True
