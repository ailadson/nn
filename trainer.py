import config
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
                weights_mats.append(
                    np.zeros(layer.weights.shape, dtype=config.FLOAT_TYPE)
                )
        return weights_mats

    def init_bias_deriv_vecs(self):
        bias_vecs = []
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights():
                bias_vecs.append(None)
            else:
                bias_vecs.append(
                    np.zeros(layer.biases.shape, dtype=config.FLOAT_TYPE)
                )
        return bias_vecs

    def train_with_examples(self, examples):
        self.reset()

        num_examples = len(examples)
        loss_sum = 0
        misclassification = 0
        for example in examples:
            loss, misclas = self.train_with_example(example)
            loss_sum += loss
            misclassification += misclas
        self.update_weights_and_biases_of_net(num_examples)
        return (loss_sum / num_examples, misclassification / num_examples)

    def train_with_example(self, example):
        ipt, expected_output = example
        output = self.net.forward_propagate(ipt)
        expected_true_class_idx = list(expected_output).index(1)

        self.net.back_propagate(expected_output)
        self.accumulate_weight_deriv_mats_and_bias_vecs()
        estimated_true_class_idx = list(output).index(max(output))

        loss = cross_entropy(output, expected_true_class_idx)
        misclassification = (
            1 if expected_true_class_idx != estimated_true_class_idx else 0
        )
        return (loss, misclassification)

    #accumulate derivative over course of batch
    def accumulate_weight_deriv_mats_and_bias_vecs(self):
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue
            # Not sure if incrementing a numpy obj inside a list creates a copy
            # of the numpy array
            temp_weight_mat = self.weight_deriv_mats[i]
            temp_weight_mat += layer.deriv_cache.weights
            temp_bias_vec = self.bias_deriv_vecs[i]
            temp_bias_vec += layer.deriv_cache.biases

    #at the end of batch, perform update
    def update_weights_and_biases_of_net(self, num_of_examples):
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue
            self.weight_deriv_mats[i] *= (-self.learning_rate / num_of_examples)
            self.bias_deriv_vecs[i] *= (-self.learning_rate / num_of_examples)
            layer.weights += self.weight_deriv_mats[i]
            layer.biases += self.bias_deriv_vecs[i]

    def reset(self):
        for i, layer in enumerate(self.net.layers):
            if layer.has_weights():
                self.weight_deriv_mats[i] *= 0
                self.bias_deriv_vecs[i] *= 0
