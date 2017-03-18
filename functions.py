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

def tanh(source, des = None):
    return np.tanh(source, out = des)

def deriv_tanh(z, des = None):
    if isinstance(z, np.ndarray):
        tanh(z, des)
        des *= des
        des *= -1
        des += 1
    else:
        return 1 - tanh(z)**2

def inner_product(left, right):
    if len(left) != len(right):
        raise "Error"
    s = 0
    for val1, val2 in zip(left, right):
        s += (val1 * val2)
    return s

def matrix_add(target, dest):
    if len(target) != len(dest) or len(target[0]) != len(dest[0]):
        raise "Matrices need to be the same shape"
    for i in range(len(target)):
        for j in range(len(target[i])):
            dest[i][j] += target[i][j]

def matrix_scale(matrix, scalar):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= scalar

def matrix_times_vec(mat, vec, output = None):
    if output is None:
        output = [0 for i in range(len(mat))]
    if len(mat[0]) != len(vec):
        raise "Error"
    for i, row in enumerate(mat):
        output[i] = inner_product(row, vec)
    return output

def cross_entropy(estimated_probs, true_class_idx):
    estimated_prob = estimated_probs[true_class_idx]
    if estimated_prob == 0: #underflow error, return high loss
        print(estimated_probs)
        return 100
    return -1 * math.log(estimated_prob)

def derivative_of_ce(estimated_prob_pos, observation):
    if observation == 1:
        return -1/estimated_prob_pos
    else:
        return 1/(1 - estimated_prob_pos)

def derivative_of_sig(z, des = None):
    if isinstance(z, np.ndarray):
        temp = np.copy(z)
        sigmoid(z, des)
        sigmoid(temp, temp)
        temp *= -1
        temp += 1
        des *= temp
    else:
        return sigmoid(z) * (1 - sigmoid(z))

def zeros(shape):
    if len(shape) == 2:
        row, col = shape
        return [[0] * col for i in range(row)]
    elif len(shape) == 1:
        return [0] * shape[0]
    else:
        raise "Bad Shape"

def zeros_like(mat):
    return zeros([len(mat), len(mat[0])])

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
