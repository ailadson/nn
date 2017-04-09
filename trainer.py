import config
from functions.activations import cross_entropy
from functions.helpers import np2str
import numpy as np

class Trainer:
    def __init__(self, net, learning_rate):
        self.net = net
        self.learning_rate = learning_rate
        self.weight_deriv_mats = self.init_weight_deriv_mats()
        self.bias_deriv_vecs = self.init_bias_deriv_vecs()

    # ==Initialization methods==
    def init_weight_deriv_mats(self):
        weights_mats = []
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights():
                weights_mats.append(None)
            else:
                weights_mats.append(
                    config.float_zeros(layer.weights.shape)
                )
        return weights_mats

    def init_bias_deriv_vecs(self):
        bias_vecs = []
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights():
                bias_vecs.append(None)
            else:
                bias_vecs.append(
                    config.float_zeros(layer.biases.shape)
                )
        return bias_vecs

    # ==Training methods==
    def reset(self):
        for i, layer in enumerate(self.net.layers):
            if layer.has_weights():
                self.weight_deriv_mats[i] *= 0
                self.bias_deriv_vecs[i] *= 0

    def train_with_examples(self, examples):
        self.reset()

        num_examples = len(examples)
        loss_sum = 0
        misclassifications = 0
        for example in examples:
            loss, did_misclassify = self.train_with_example(example)
            loss_sum += loss
            if did_misclassify: misclassifications += 1

        self.update_weights_and_biases_of_net(num_examples)

        avg_loss = loss_sum / num_examples
        misclassification_rate = misclassifications / num_examples
        return (avg_loss, misclassification_rate)

    def train_with_example(self, example):
        input_v, expected_output = example
        expected_true_class_idx = list(expected_output).index(1)

        if config.DEBUG_STEP_BY_STEP:
            print(">>> Forward Propagate <<<")
        output = self.net.forward_propagate(input_v)
        if config.DEBUG_LOG_ACTIVATIONS:
            self.log_activations()

        if config.DEBUG_STEP_BY_STEP:
            print(">>> Back Propagate <<<")
        self.net.back_propagate(expected_output)
        if config.DEBUG_LOG_DERIVATIVES:
            self.log_derivatives()
        self.accumulate_weight_deriv_mats_and_bias_vecs()

        estimated_true_class_idx = list(output).index(max(output))
        # TODO: put me back in.
        loss = 0.0 #cross_entropy(output, expected_true_class_idx)
        did_misclassify = (
            expected_true_class_idx != estimated_true_class_idx
        )

        if config.DEBUG_STEP_BY_STEP:
            input("press enter to continue")

        return (loss, did_misclassify)

    # ==Update methods==

    # accumulate derivative over course of batch
    def accumulate_weight_deriv_mats_and_bias_vecs(self):
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue
            # Not sure if incrementing a numpy obj inside a list
            # creates a copy of the numpy array
            temp_weight_mat = self.weight_deriv_mats[i]
            temp_weight_mat += layer.deriv_wrt_weights()
            temp_bias_vec = self.bias_deriv_vecs[i]
            temp_bias_vec += layer.deriv_wrt_biases()

    # at the end of batch, perform update
    def update_weights_and_biases_of_net(self, num_examples):
        learning_rate = (-self.learning_rate / num_examples)
        for i, layer in enumerate(self.net.layers):
            if not layer.has_weights(): continue

            self.weight_deriv_mats[i] *= learning_rate
            self.bias_deriv_vecs[i] *= learning_rate

            layer.weights += self.weight_deriv_mats[i]
            layer.biases += self.bias_deriv_vecs[i]

    # == Logging Methods ==
    def log_activations(self):
        for layer_idx, layer in enumerate(self.net.layers):
            if layer_idx == 0: continue
            print(f"Layer #{layer_idx} {type(layer)} | "
                  f"activations: "
                  f"{np2str(layer.output)}")

        logits = self.net.layers[-1].logits()
        print(f"Output logits: {np2str(logits)}")

    def log_derivatives(self):
        for layer_idx in reversed(range(len(self.net.layers))):
            layer = self.net.layers[layer_idx]
            if not layer.has_weights(): continue

            print(f"Layer #{layer_idx} {type(layer)} | "
                  f"deriv_wrt_weights: "
                  f"{np2str(layer.deriv_wrt_weights())}")
