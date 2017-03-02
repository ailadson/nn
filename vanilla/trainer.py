from functions import *

class Trainer:
    def __init__(self, net, learning_rate):
        self.net = net
        self.learning_rate = learning_rate
        self.weight_deriv_mats = self.init_weight_deriv_mats()

    def init_weight_deriv_mats(self):
        weights_mats = []
        for i, layer in enumerate(self.net.layers):
            if i == 0:
                weights_mats.append(None)
            else:
                weights_mats.append( zeros_like(layer.weights) )
        return weights_mats

    def train_with_examples(self,examples):
        loss_sum = 0
        misclassification = 0
        for example in examples:
            loss, misclas = self.train_with_example(example)
            loss_sum += loss
            misclassification += misclas
        self.update_weights(len(examples))
        return ( loss_sum/len(examples), misclassification/len(examples) )

    def train_with_example(self, example):
        ipt, opt = example
        output = self.net.forward_propagate(ipt)[0] #unwrapping
        self.net.back_propagate(opt)
        self.update_weight_deriv_mats()
        loss = cross_entropy(output, opt)
        guess = 1 if output >= 0.5 else 0
        misclassification = 1 if opt != guess else 0
        return (loss, misclassification)

    def update_weight_deriv_mats(self):
        for i, layer in enumerate(self.net.layers):
            if i == 0: continue
            matrix_add(layer.deriv_cache.weights, self.weight_deriv_mats[i])

    def update_weights(self, num_of_examples):
        for i, layer in enumerate(self.net.layers):
            if i == 0: continue
            matrix_scale(self.weight_deriv_mats[i], -self.learning_rate / num_of_examples)
            matrix_add(self.weight_deriv_mats[i], layer.weights)
