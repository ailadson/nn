from functions import *
import numpy as np


class SimpleTrainer:
    def __init__(self, net, learning_rate):
        self.net = net
        self.learning_rate = learning_rate
        self.weight_deriv_mats = self.init_weight_deriv_mats()
        # self.bias_deriv_vecs = self.init_bias_deriv_vecs()

    def init_weight_deriv_mats(self):
        weights_mats = []
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights():
                weights_mats.append(None)
            else:
                weights_mats.append( np.zeros(layer.weights.shape) , dtype=np.float32)
        return weights_mats

    def init_bias_deriv_vecs(self):
        bias_vecs = []
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights():
                bias_vecs.append(None)
            else:
                bias_vecs.append( np.zeros(layer.biases.shape) , dtype=np.float32)
        return bias_vecs

    def train_with_examples(self, examples):
        for example in examples:
            self.train_with_example(example)
        self.update_weights_and_biases(len(examples))

    def train_with_example(self, example):
        ipt, opt = example
        output = self.net.forward_propagate(ipt)
        self.net.back_propagate(opt)
        self.update_weight_deriv_mats_and_bias_vecs()

    #accumulate derivative over course of batch
    def update_weight_deriv_mats_and_bias_vecs(self):
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue
            self.weight_deriv_mats[i] += layer.deriv_cache.weights
            # self.bias_deriv_vecs[i] += layer.deriv_cache.unit_total_inputs

    #at the end of batch, perform update
    def update_weights_and_biases(self, num_of_examples):
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue
            self.weight_deriv_mats[i] *= (-self.learning_rate / num_of_examples)
            # self.bias_deriv_vecs[i] *= (-self.learning_rate / num_of_examples)
            layer.weights += self.weight_deriv_mats[i]
            # layer.biases += self.bias_deriv_vecs[i]
