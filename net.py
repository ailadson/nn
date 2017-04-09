from layers.convolution_layer import ConvolutionLayer
from layers.flatten_layer import FlattenLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.input_layer import InputLayer
from layers.max_pooling_layer import MaxPoolingLayer
from layers.rank3input_layer import Rank3InputLayer
from layers.softmax_layer import SoftmaxLayer

class Net:
    def __init__(self):
        self.layers =[]

    # Builder methods
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

    def add_conv_layer(
            self,
            num_out_layers,
            kernel_height,
            kernel_width,
            activation_fn_name):

        l = ConvolutionLayer(
            self.layers[-1],
            num_out_layers,
            kernel_height,
            kernel_width,
            activation_fn_name
        )
        self.layers.append(l)

    def add_softmax_layer(self):
        l = SoftmaxLayer(self.layers[-1])
        self.layers.append(l)

    # Other
    def back_propagate(self, expected_output):
        self.layers[-1].set_observed_output(expected_output)
        for layer in reversed(self.layers):
            if layer.has_weights():
                layer.deriv_wrt_weights()
                layer.deriv_wrt_biases()

    def forward_propagate(self, input_v):
        self.reset()
        self.layers[0].set_input(input_v)
        return self.layers[-1].outputs()

    def reset(self):
        for layer in self.layers:
            layer.reset_caches()
