import cupy as cp
from cupyx.profiler import benchmark

def my_func(a):
    return cp.sqrt(cp.sum(a**2, axis=-1))

a = cp.random.random((256, 1024))

print(benchmark(my_func, (a,), n_repeat=20))
