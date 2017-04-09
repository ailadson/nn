import config
from layers.layer import Layer
import numpy as np

class Rank3InputLayer(Layer):
    def __init__(self, shape):
        super().__init__(None, shape, None)

    def set_input(self, input_v):
        if input_v.shape != self.output_shape:
            raise Exception("Bad shape")

        np.copyto(self.activation_cache.outputs, input_v)
        self.activation_cache.set("outputs")

    def has_weights(self):
        return False
