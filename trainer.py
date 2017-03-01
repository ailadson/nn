from functions import *

class Trainer:
    def __init__(self, net, learning_rate):
        self.net = net
        self.learning_rate = learning_rate
        self.weight_deriv_mats = self.init_weight_deriv_mats()

    def init_weight_deriv_mats(self):
        weights_mats = []
        for layer in self.net.layers:
            weights_mats.append( zeros_like(layer.weights) )
        return weights_mats

    def train_with_examples(self,examples):
        for example in examples:
            self.train_with_examples(example)
        self.update_weights(len(examples))

    def train_with_example(self, example):
        ipt, opt = example
        net.forward_propagate(ipt)
        net.back_propagate(opt)
        self.update_weight_deriv_mats()

    def update_weight_deriv_mats(self):
        for i, layer in enumerate(self.net.layers):
            matrix_add(layer.deriv_cache.weights, self.weight_deriv_mats[i])

    def update_weights(self, num_of_examples):
        for i, layer in enumerate(self.net.layers):
            matrix_scale(self.weight_deriv_mats[i], -self.learning_rate / num_of_examples)
            matrix_add(self.weight_deriv_mats[i], layer.weights)
