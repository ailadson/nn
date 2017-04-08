from itertools import *
import numpy as np
import pyx.avx_convolve2d as convolve2d
import pyx.deconvolve2d as deconvolve2d
import pyx.max_pooling_functions as max_pooling_functions

IMG_DIM = 28
def make_test_input(dim = IMG_DIM):
    x = np.zeros((1, dim, dim)).astype(np.float32)

    for (i, j) in product(range(dim), range(dim)):
        x[0, i, j] = i + j

    return x

def make_multi_channel_input(num_channels, dim = IMG_DIM):
    x = np.zeros((num_channels, dim, dim)).astype(np.float32)

    for (k, i, j) in product(range(num_channels), range(dim), range(dim)):
        x[k, i, j] = k + i + j

    return x

def make_blank_multi_channel_input(num_channels, dim = IMG_DIM):
    x = np.zeros((num_channels, dim, dim)).astype(np.float32)
    return x

KERNEL_DIM = 3
def make_identity_kernel():
    k = np.zeros((1, 1, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    k[0, 0, KERNEL_DIM // 2, KERNEL_DIM // 2] = 1.0
    return k

def make_blank_kernel():
    k = np.zeros((1, 1, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    return k

def make_nd_kernel(num_of_dim):
    k = np.zeros((1, num_of_dim, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    k[0, 0, 1, 1] = 1.0
    k[0, 1, 0, 0] = 1.0
    k[0, 1, 0, 2] = 1.0
    k[0, 1, 2, 0] = 1.0
    k[0, 1, 2, 2] = 1.0
    return k

def make_fancy_kernel():
    k = np.zeros((1, 1, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    k.fill(1.0)
    k[0, 0, 1, 1] = 0.0
    return k

def make_empty_output(dim = IMG_DIM):
    y = np.zeros((1, dim, dim)).astype(np.float32)
    return y

def make_multi_channel_input_expected_output():
    return np.array([[[3.,  7., 5.],
                      [7., 14., 9.],
                      [5.,  9., 7.]]])

def expected_fancy_kernel_output():
    y = np.zeros((1, IMG_DIM, IMG_DIM)).astype(np.float32)
    for (i, j) in product(range(IMG_DIM), range(IMG_DIM)):
        val = 0.0
        if i > 0:
            if j > 0:
                val += (i - 1) + (j - 1)
            if j < (IMG_DIM - 1):
                val += (i - 1) + (j + 1)
            val += (i - 1) + j
        if i < (IMG_DIM - 1):
            if j > 0:
                val += (i + 1) + (j - 1)
            if j < (IMG_DIM - 1):
                val += (i + 1) + (j + 1)
            val += (i + 1) + j
        if j > 0:
            val += i + (j - 1)
        if j < (IMG_DIM - 1):
            val += i + (j + 1)

        y[0, i, j] = val

    return y

def make_simple_asym_kernel():
    k = np.zeros((1, 1, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    k[0, 0, 0, 0] = 1.0
    return k

def expected_simple_backward_conv_output():
    y = np.zeros((1, IMG_DIM, IMG_DIM)).astype(np.float32)
    for (i, j) in product(range(IMG_DIM), range(IMG_DIM)):
        if i == IMG_DIM - 1 or j == IMG_DIM - 1:
            continue
        y[0, i, j] = (i + 1) + (j + 1)
    return y

def naive_deconvolve(input, kernel, output):
    for (kernel_i, kernel_j) in product(range(KERNEL_DIM), range(KERNEL_DIM)):
        kernel_offset_i = kernel_i - KERNEL_DIM // 2
        kernel_offset_j = kernel_j - KERNEL_DIM // 2
        for (input_i, input_j) in product(range(IMG_DIM), range(IMG_DIM)):
            out_i = input_i + kernel_offset_i
            out_j = input_j + kernel_offset_j
            if out_i < 0 or out_i >= IMG_DIM or out_j < 0 or out_j >= IMG_DIM:
                continue
            kernel[0, 0, kernel_i, kernel_j] += (
                output[0, out_i, out_j] * input[0, input_i, input_j]
            )

def test_identity_kernel():
    x = make_test_input()
    k = make_identity_kernel()
    y = make_empty_output()
    convolve2d.apply_convolution(x, k, y)
    assert (x == y).all()
    print("Identity kernel test passed!")

def test_fancy_kernel():
    x = make_test_input()
    k = make_fancy_kernel()
    y = make_empty_output()
    convolve2d.apply_convolution(x, k, y)
    assert (y == expected_fancy_kernel_output()).all()
    print("Fancy kernel test passed!")

def test_backward_convolve():
    x = make_test_input()
    k = make_simple_asym_kernel()
    y = make_empty_output()
    convolve2d.apply_backward_convolution(x, k, y)
    assert (y == expected_simple_backward_conv_output()).all()
    print("Backward kernel test passed!")

def test_deconvolve():
    x = make_test_input()
    y = make_test_input() + 1
    k = make_blank_kernel()

    expected_k = make_blank_kernel()
    naive_deconvolve(x, expected_k, y)

    deconvolve2d.deriv_wrt_weights(x, k, y)
    assert (k == expected_k).all()
    print("Deconvolve2d test passed!")

def test_multi_channel_input_convolve():
    x = make_multi_channel_input(3, 3)
    k = make_nd_kernel(3)
    y = make_blank_multi_channel_input(1, 3)

    convolve2d.apply_convolution(x, k, y)
    expected_result = make_multi_channel_input_expected_output()

    assert (y == expected_result).all()
    print("Multichannel Input passed!")

HALF_DIM = IMG_DIM // 2
def test_apply_max_pooling():
    x = make_test_input()
    y = make_empty_output(HALF_DIM)
    max_pooling_functions.apply_max_pooling(x, y)

    expected_result = make_empty_output(HALF_DIM)
    for (i, j) in product(range(HALF_DIM), range(HALF_DIM)):
        expected_result[0, i, j] = (2*i + 1) + (2*j + 1)

    assert (y == expected_result).all()
    print("apply_max_pooling test passed!")

def test_back_propagate_channels():
    deriv_wrt_prev_outputs = make_empty_output()
    prev_channels = make_test_input()
    deriv_wrt_unit_outputs = make_test_input(HALF_DIM) * 10
    max_pooling_functions.back_propagate_channels(
        deriv_wrt_prev_outputs,
        prev_channels,
        deriv_wrt_unit_outputs
    )

    expected_result = np.zeros((1, IMG_DIM, IMG_DIM))
    for (i, j) in product(range(IMG_DIM), range(IMG_DIM)):
        if (i % 2 == 0) or (j % 2 == 0): continue
        expected_result[0, i, j] = 10 * (i // 2 + j // 2)

    assert (deriv_wrt_prev_outputs == expected_result).all()
    print("back_propagate_channel test passed!")

test_identity_kernel()
test_fancy_kernel()
test_backward_convolve()
test_deconvolve()
test_multi_channel_input_convolve()
test_apply_max_pooling()
test_back_propagate_channels()
