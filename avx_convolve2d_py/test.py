import main
import time

t = time.time()
for i in range(1_000_000):
    val = main.run_test()
print(val)
print(time.time() - t)
