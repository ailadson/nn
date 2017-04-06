import avx_convolve2d_main_py
import time

t = time.time()
for i in range(8_000_000):
    val = avx_convolve2d_main_py.main()
print(val)
print(time.time() - t)
