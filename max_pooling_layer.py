from functions import *
from deriv_cache import MaxPoolDerivativeCache
from max_pooling_functions import *
import skimage.measure
import numpy as np
import math


class MaxPoolingLayer():
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        prev_shape = prev_layer.output.shape
        self.num_of_input_layers = prev_shape[0]
        self.output = np.zeros([prev_shape[0], math.ceil(prev_shape[1]/2), math.ceil(prev_shape[2]/2)])
        self.activation_func = lambda val, des: np.copyto(des, val)
        self.deriv_activation_func = lambda val, des: des.fill(1)
        self.total_input = np.zeros(self.output.shape)
        self.deriv_cache = MaxPoolDerivativeCache(self)

    def forward_propagate(self):
        self.total_input *= 0;
        self.apply_pooling(self.prev_layer.output, self.total_input)
        self.activation_func(self.total_input, self.output)

    def back_propagate(self):
        self.deriv_cache.reset()
        self.deriv_wrt_prev_outputs()
        # print("Print Derivative Weights")
        # print(self.deriv_cache.weights)

    def deriv_wrt_unit_outputs(self):
        return self.next_layer.deriv_wrt_prev_outputs()

    def deriv_wrt_prev_outputs(self):
        if self.deriv_cache.is_set("prev_outputs"):
            return self.deriv_cache.prev_outputs

        self.deriv_cache.prev_outputs *= 0

        back_propagate_channels(self.deriv_cache.prev_outputs, self.prev_layer.output, self.deriv_wrt_unit_outputs())
        # for channel_idx in range(self.num_of_input_layers):
        #     self.back_propagate_channel(channel_idx)
        self.deriv_cache.set('prev_outputs')
        return self.deriv_cache.prev_outputs

    def back_propagate_channel(self, channel_idx):
        back_propagate_channel(self, channel_idx)

    #     derive_wrt_channel_output = self.deriv_wrt_unit_outputs()[channel_idx]
    #     channel = self.prev_layer.output[channel_idx]
    #     for i in range(derive_wrt_channel_output.shape[0]):
    #         for j in range(derive_wrt_channel_output.shape[1]):
    #             values = self.getpoolingvalues(channel, i, j)
    #             max_i, max_j = self.get_local_max_idx(i, j, values)
    #             self.deriv_cache.prev_outputs[channel_idx, max_i, max_j] = derive_wrt_channel_output[i, j]
    #
    # def get_local_max_idx(self, block_i, block_j, values):
        # get_local_max_idx(block_i, block_j, values)
    #     i = block_i * 2
    #     j = block_j * 2
    #     n1, n2, n3, n4 = values
    #     local_max = max(values)
    #     if n1 == local_max:
    #         return (i,j)
    #     elif n2 == local_max:
    #         return (i,j+1)
    #     elif n3 == local_max:
    #         return (i+1,j)
    #     elif n4 == local_max:
    #         return (i+1,j+1)

    # def getpoolingvalues(self, channel, block_i, block_j):
    #     getpoolingvalues(channel, block_i, block_j)
    #     i = block_i * 2
    #     j = block_j * 2
    #     good_i, good_j = (i + 1 < channel.shape[0], j + 1 < channel.shape[1])
    #     n1 = channel[i, j]
    #     n2 = channel[i, j + 1] if good_j else -math.inf
    #     n3 = channel[i + 1, j] if good_i else -math.inf
    #     n4 = channel[i + 1, j + 1] if good_i and good_j else -math.inf
    #     return (n1, n2, n3, n4)



    def apply_pooling(self, ipt, des):
        np.copyto(des, skimage.measure.block_reduce(ipt, (1,2,2), np.max))

    def has_weights(self):
        return False
