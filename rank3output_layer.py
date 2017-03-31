from deriv_cache import ConvDerivativeCache
from functions import *
import math

class Rank3OutputLayer():
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        prev_layer.next_layer = self
        self.weights = np.zeros(2)
        self.total_input = np.zeros(2)
        self.output = self.prev_layer.output
        self.observed_output = None
        self.deriv_cache = ConvDerivativeCache(self)

    def forward_propagate(self):
        pass
    #     # super().forward_propagate()
    #     # outputs is in logits. make greatest value 0
    #     self.output -= max(self.output)
    #     # before softmax, hold on to logits
    #     np.copyto(self.logits , self.output)
    #     softmax(self.output, self.output)

    def back_propagate(self):
        pass

    def set_observed_output(self, observed):
        self.observed_output = observed

    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs

        # manually doing derivative of L2 norm
        np.copyto(self.deriv_cache.prev_outputs, self.observed_output)
        self.deriv_cache.prev_outputs -= self.prev_layer.output
        self.deriv_cache.prev_outputs *= -2
        # normalize
        self.deriv_cache.prev_outputs /= (
            math.sqrt(((self.observed_output - self.prev_layer.output) ** 2).sum())
        )
        # print("Print Prev Output")
        # print(self.deriv_cache.prev_outputs)
        # print(self.prev_layer.output)
        return self.deriv_cache.prev_outputs

    def has_weights(self):
        return False
