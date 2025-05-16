from timeit import timeit
from numba import vectorize, float64, jit
import numpy as np

N = int(100)

# comment in the one to be tested
# @vectorize([float64(float64, float64)])
# @jit
def collect_input(x, y):
    tmp = x
    for i in range(N):
        tmp += y
    return tmp

# comment in if tested
# @jit
def learn():
    for i in range(N):
        collect_input(np.ones(i), np.arange(i))

# f(np.zeros(0), np.zeros(0)), g()

time = timeit(lambda: collect_input(np.ones(N), np.arange(N)), number=N)
print(f"Done. time (f)={time:.4f}")

time = timeit(learn, number=N)
print(f"Done. time (g)={time:.4f}")


"""
c = collect_data()
l = learn()

               |  time(c) (N = 100) |      time(l)      |  time(c) (N = 2000)
---------------+--------------------+-------------------+---------------------
vanilla        |       0.0070       |      0.6089       |     6.6778
c vec          |       0.0007       |      0.0470       |     7.0914
c jit          |   0.1347 (0.0009)  |      0.0443       |  4.4007 (4.2416)  
c vec + l jit  |       0.0007       |  0.3061 (0.0218)  |      ----
c jit + l jit  |   0.1289 (0.0009)  |  0.2743 (0.0287)  |      ----

in brackets time with pre-compiled function (line with initial "empty" calls of collect_data and learn commented in) 

Explanations:

- vectorization results in about a 10x speed-up compared to vanilla python code for *collect_data*; jit-compiled *collect_data* seems to be on the other hand 100x slower compared to pure python. *learn* sees the same slow-down

- if the initial calls of *learn* and *collect_data* are commented in, all jit-compiled functions show about the same speed-up as the vectorized ones. The initially observed slow-down was due to the compilation process having been implicitly being part of the benchmarking (as the functions are compiled on their first call). This overhead can be significant and should be taken into account when choosing the method.

- for N=2000, the jit-compiled code show still a significant speed-up with respect to pure python, but only by a factor of about 1.5. The overhead also becomes more negligible. The vectorized version on the other hand is now even a slightly bit slower than the pure python code. This is due to the fact that even the pure python code is already optimized for the numpy array addition within *collect_input* and the implementation numpy uses an optimised code for very long arrays.

"""
