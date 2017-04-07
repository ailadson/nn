import pyx.avx_convolve2d_py as avx_convolve2d_py
import time

t = time.time()
for i in range(1_000_000):
    val = avx_convolve2d_py.run_test()
print(val)
print(time.time() - t)
