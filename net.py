from input_layer import InputLayer
from output_layer import OutputLayer
from layer import FullyConnectedLayer
from conv_layer import ConvolutionalLayer
from rank3input_layer import Rank3InputLayer
from rank3output_layer import Rank3OutputLayer
from max_pooling_layer import MaxPoolingLayer
from flatten_layer import FlattenLayer

class Net:
    def __init__(self):
        self.layers =[]

    def add_input_layer(self, num_units):
        self.layers.append(InputLayer(num_units))

    def add_rank3_input_layer(self, shape):
        self.layers.append(Rank3InputLayer(shape))

    def add_fc_layer(self, num_units):
        l = FullyConnectedLayer(self.layers[-1], num_units)
        self.layers.append(l)

    def add_flatten_layer(self):
        l = FlattenLayer(self.layers[-1])
        self.layers.append(l)

    def add_max_pool_layer(self):
        l = MaxPoolingLayer(self.layers[-1])
        self.layers.append(l)

    def add_conv_layer(self, height, width, num_of_output_layers):
        l = ConvolutionalLayer(self.layers[-1], height, width, num_of_output_layers)
        self.layers.append(l)

    def add_rank3output_layer(self):
        l = Rank3OutputLayer(self.layers[-1])
        self.layers.append(l)

    def add_output_layer(self, num_units):
        l = OutputLayer(self.layers[-1], num_units)
        self.layers.append(l)

    def forward_propagate(self, input_v):
        self.layers[0].set_input(input_v)

        for layer in self.layers:
            layer.forward_propagate()

        return self.layers[-1].output

    def back_propagate(self, observed_output):
        self.layers[-1].set_observed_output(observed_output)

        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i]
            layer.back_propagate()
