import math
import numpy as np

def sigmoid(source, des = None):
    if isinstance(source, np.ndarray):
        np.copyto(des, source)
        des *= -1
        np.exp(des, out = des)
        des += 1
        np.reciprocal(des, out = des)
    else:
        return 1/(1+math.exp(-source))

def derivative_of_sigmoid(z, des = None):
    if isinstance(z, np.ndarray):
        temp = np.copy(z)
        sigmoid(z, des)
        sigmoid(temp, temp)
        temp *= -1
        temp += 1
        des *= temp
    else:
        return sigmoid(z) * (1 - sigmoid(z))

def tanh(source, des = None):
    return np.tanh(source, out = des)

def derivative_of_tanh(z, des = None):
    if isinstance(z, np.ndarray):
        tanh(z, des)
        des *= des
        des *= -1
        des += 1
    else:
        return 1 - tanh(z)**2

def cross_entropy(estimated_probs, true_class_idx):
    estimated_prob = estimated_probs[true_class_idx]
    if estimated_prob == 0: #underflow error, return high loss
        print("CE Error?")
        print(estimated_probs)
        raise "Hey"
        return 100
    return -1 * math.log(estimated_prob)

def derivative_of_ce(estimated_prob_pos, observation, out = None):
    if isinstance(estimated_prob_pos, np.ndarray):
        out.fill(0)
        out += ((-1 / estimated_prob_pos) * observation)
        out += ((1 / (1 - estimated_prob_pos)) * (1 - observation))
        return out

    if observation == 1:
        return -1/estimated_prob_pos
    else:
        return 1/(1 - estimated_prob_pos)

def softmax(values, des = None):
    if des is not None:
        np.exp(values, des)
        sum_e = sum(des)
        if math.isnan(sum_e):
            print(values)
            print(des)
            raise Exception()
        des /= sum_e
    else:
        raise Exception("Not implemented!!!!!!!!!")

def derivative_of_softmax_and_ce(logits, true_class_idx, des):
    np.exp(logits, out = des)
    sum_of_exp_logits = sum(des)
    des /= sum_of_exp_logits
    des[true_class_idx] -= 1

def relu(values, des = None):
    if des is not None:
        np.copyto(des, np.maximum(values, 0))
    else:
        return np.maximum(values, 0)

def derivative_of_relu(values, des = None):
    if des is not None:
        des.fill(1)
        des *= (values > 0)
    else:
        raise Exception("Not implemented")
