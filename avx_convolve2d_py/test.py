import main
import time

t = time.time()
for i in range(8_000_000):
    val = main.main()
print(val)
print(time.time() - t)
