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
            if not layer.has_weights():
                weights_mats.append(None)
            else:
                weights_mats.append( np.zeros(layer.weights.shape) )
        return weights_mats

    def init_bias_deriv_vecs(self):
        bias_vecs = []
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights():
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

    def inner_update_weight_deriv_mats_and_bias_vecs1(self, layer, i):
        temp_weight_mat = self.weight_deriv_mats[i]
        temp_weight_mat += layer.deriv_cache.weights


    def inner_update_weight_deriv_mats_and_bias_vecs2(self, layer, i):
        temp_bias_vec = self.bias_deriv_vecs[i]
        temp_bias_vec += layer.deriv_cache.biases

    #accumulate derivative over course of batch
    def update_weight_deriv_mats_and_bias_vecs(self):
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue
            self.inner_update_weight_deriv_mats_and_bias_vecs1(layer, i)
            self.inner_update_weight_deriv_mats_and_bias_vecs2(layer, i)
            # print("self.weights")
            # print(self.weight_deriv_mats[i].flags.f_contiguous)
            # print("layer.weights")
            # print(layer.deriv_cache.weights.flags.f_contiguous)
            # print("self.biases")
            # print(self.bias_deriv_vecs[i].flags.f_contiguous)
            # print("layer.biases")
            # print(layer.deriv_cache.biases.flags.f_contiguous)
            # print("-"*88)

    #at the end of batch, perform update
    def update_weights_and_biases(self, num_of_examples):
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue
            self.weight_deriv_mats[i] *= (-self.learning_rate / num_of_examples)
            self.bias_deriv_vecs[i] *= (-self.learning_rate / num_of_examples)
            layer.weights += self.weight_deriv_mats[i]
            layer.biases += self.bias_deriv_vecs[i]
