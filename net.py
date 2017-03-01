from input_layer import InputLayer
from layer import Layer

class Net:
    def __init__(self, num_units):
        self.layers =[InputLayer(num_units)]

    def add_layer(self, num_units):
        l = Layer(self.layers[-1], num_units)
        self.layers.append(l)

    def forward_propagate(self, input_v):
        self.layers[0].set_input(input_v)

        for layer in self.layers:
            layer.forward_propagate()

        return self.layers[-1].output

    def back_propagate(self, observed_output):
        self.layers[-1].set_observed_output(observed_output)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer.back_propagate()
