from functions import *
import numpy as np


class Trainer:
    def __init__(self, net, learning_rate):
        self.net = net
        self.learning_rate = learning_rate
        self.weight_deriv_mats = self.init_weight_deriv_mats()
        self.bias_deriv_vecs = self.init_bias_deriv_vecs()

    def init_weight_deriv_mats(self):
        weights_mats = []
        for i, layer in enumerate(self.net.layers):
            if i == 0:
                weights_mats.append(None)
            else:
                weights_mats.append( np.zeros(layer.weights.shape) )
        return weights_mats

    def init_bias_deriv_vecs(self):
        bias_vecs = []
        for i, layer in enumerate(self.net.layers):
            if i == 0:
                bias_vecs.append(None)
            else:
                bias_vecs.append( np.zeros(layer.biases.shape) )
        return bias_vecs

    def train_with_examples(self,examples):
        loss_sum = 0
        misclassification = 0
        for example in examples:
            loss, misclas = self.train_with_example(example)
            loss_sum += loss
            misclassification += misclas
        self.update_weights_and_biases(len(examples))
        return ( loss_sum/len(examples), misclassification/len(examples) )

    def train_with_example(self, example):
        ipt, opt = example
        output = self.net.forward_propagate(ipt)
        self.net.back_propagate(opt)
        self.update_weight_deriv_mats_and_bias_vecs()
        loss = cross_entropy(output, list(opt).index(1))
        estimated_true_class_idx = list(output).index( max(output) )
        misclassification = 1 if list(opt).index( 1 ) != estimated_true_class_idx else 0
        return (loss, misclassification)

    #accumulate derivative over course of batch
    def update_weight_deriv_mats_and_bias_vecs(self):
        for i, layer in enumerate(self.net.layers):
            if i == 0: continue
            self.weight_deriv_mats[i] += layer.deriv_cache.weights
            self.bias_deriv_vecs[i] += layer.deriv_cache.unit_total_inputs

    #at the end of batch, perform update
    def update_weights_and_biases(self, num_of_examples):
        for i, layer in enumerate(self.net.layers):
            if i == 0: continue
            self.weight_deriv_mats[i] *= (-self.learning_rate / num_of_examples)
            self.bias_deriv_vecs[i] *= (-self.learning_rate / num_of_examples)
            layer.weights += self.weight_deriv_mats[i]
            layer.biases += self.bias_deriv_vecs[i]
