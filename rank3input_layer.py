import numpy as np

class Rank3Input():
    def __init__(self, shape):
        self.shape = shape
        self.output = np.zeros(shape)

    def set_input(self, ipt):
        if ipt.shape != self.shape:
            raise Exception("Bad shape")
        np.copyto(self.output, ipt)

    def forward_propagate():
        pass
