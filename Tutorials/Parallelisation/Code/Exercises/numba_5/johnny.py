from timeit import timeit
import numpy as np

N = int(100)

def collect_input(x, y):
    tmp = x
    for i in range(N):
        tmp += y
    return tmp

def learn():
    for i in range(N):
        collect_input(np.ones(i), np.arange(i))

# f(np.zeros(0), np.zeros(0)), g()

time = timeit(lambda: collect_input(np.ones(N), np.arange(N)), number=N)
print(f"Done. time (f)={time:.4f}")

time = timeit(learn, number=N)
print(f"Done. time (g)={time:.4f}")
