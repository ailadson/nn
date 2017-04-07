import numpy as np

class Rank3InputLayer():
    def __init__(self, shape):
        self.shape = shape
        self.output = np.zeros(shape, dtype=np.float32)

    def set_input(self, ipt):
        if ipt.shape != self.shape:
            raise Exception("Bad shape")
        np.copyto(self.output, ipt)

    def forward_propagate(self):
        pass

    def has_weights(self):
        return False
