import math

def sigmoid(source, des = None):
    if isinstance(source, list):
        for i, val in enumerate(source):
            des[i] = 1/(1+math.exp(-val))
    else:
        return 1/(1+math.exp(-x))

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

def cross_entropy(estimated_prob_pos, observation):
    if observation == 1:
        estimated_prob = estimated_prob_pos
    else:
        estimated_prob = 1 - estimated_prob
    return -1 * math.log(estimated_prob)

def derivative_of_ce(estimated_prob_pos, observation):
    if observation == 1:
        return 1/(1 - estimated_prob_pos)
    else:
        return -1/estimated_prob

def derivative_of_sig(z):
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
