from itertools import *
import numpy as np
import pyx.avx_convolve2d as convolve2d
import pyx.deconvolve2d as deconvolve2d

IMG_DIM = 16
def make_test_input():
    x = np.zeros((1, IMG_DIM, IMG_DIM)).astype(np.float32)

    for (i, j) in product(range(IMG_DIM), range(IMG_DIM)):
        x[0, i, j] = i + j

    return x

KERNEL_DIM = 3
def make_identity_kernel():
    k = np.zeros((1, 1, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    k[0, 0, KERNEL_DIM // 2, KERNEL_DIM // 2] = 1.0
    return k

def make_blank_kernel():
    k = np.zeros((1, 1, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    return k

def make_fancy_kernel():
    k = np.zeros((1, 1, KERNEL_DIM, KERNEL_DIM)).astype(np.float32)
    k.fill(1.0)
    k[0, 0, 1, 1] = 0.0
    return k

def make_empty_output():
    y = np.zeros((1, IMG_DIM, IMG_DIM)).astype(np.float32)
    return y

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

test_identity_kernel()
test_fancy_kernel()
test_backward_convolve()
test_deconvolve()
