from layer.input_layer import InputLayer
from layer.softmax_layer import SoftmaxLayer
from layer.fully_connected_layer import FullyConnectedLayer
from layer.convolution_layer import ConvolutionalLayer
from layer.rank3input_layer import Rank3InputLayer
from layer.max_pooling_layer import MaxPoolingLayer
from layer.flatten_layer import FlattenLayer

class Net:
    def __init__(self):
        self.layers =[]

    def add_input_layer(self, num_units):
        self.layers.append(InputLayer(num_units))

    def add_rank3_input_layer(self, shape):
        self.layers.append(Rank3InputLayer(shape))

    def add_fc_layer(self, num_units, activation_fn_name):
        l = FullyConnectedLayer(self.layers[-1], num_units, activation_fn_name)
        self.layers.append(l)

    def add_flatten_layer(self):
        l = FlattenLayer(self.layers[-1])
        self.layers.append(l)

    def add_max_pool_layer(self):
        l = MaxPoolingLayer(self.layers[-1])
        self.layers.append(l)

    def add_conv_layer(self, kernel_shape, activation_fn_name):
        l = ConvolutionalLayer(self.layers[-1], kernel_shape, activation_fn_name)
        self.layers.append(l)

    def add_softmax_layer(self):
        l = SoftmaxLayer(self.layers[-1])
        self.layers.append(l)

    def back_propagate(self, observed_output):
        self.layers[-1].set_observed_output(observed_output)

        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i]
            layer.back_propagate()

    def forward_propagate(self, input_v):
        self.layers[0].set_input(input_v)

        for layer in self.layers:
            layer.forward_propagate()

        return self.layers[-1].output
