import pyx.avx_convolve2d as convolve2d
import pyx.deconvolve2d as deconvolve2d
import numpy as np
import time

NUM_ITERS = 1_000

x = np.random.uniform(size = (20, 28, 28)).astype(np.float32)
k = np.random.uniform(size = (40, 20, 3, 3)).astype(np.float32)
y = np.zeros((40, 28, 28), dtype = np.float32)

def run_test(name, fn):
    start_time = time.time()
    for i in range(NUM_ITERS):
        fn()
    end_time = time.time()
    iters_per_sec = NUM_ITERS / (end_time - start_time)
    print(f"{name}: {iters_per_sec:.1f} ex/sec")


run_test(
    "apply_convolution",
    lambda: convolve2d.apply_convolution(x, k, y)
)

run_test(
    "apply_backward_convolution",
    lambda: convolve2d.apply_backward_convolution(y, k, x)
)

run_test(
    "deriv_wrt_weights",
    lambda: deconvolve2d.deriv_wrt_weights(x, k, y)
)
