from PIL.Image import preinit

from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
import sklearn.metrics as sk
import timeit as it
import numpy as np

np.random.seed(0)
n = 100
predicted = np.linspace(-10,10,n)
observed = predicted + np.random.normal(0,1.5,n)

#calcuate MSE
calcs = 1000

mse_vanilla = round(vanilla_mse(observed,predicted),5)
time_vanilla = it.timeit('vanilla_mse(observed,predicted)', globals=globals(), number=calcs)
#print(time_vanilla)
mse_numpy = round(numpy_mse(observed,predicted),5)
time_np = it.timeit('numpy_mse(observed,predicted)', globals=globals(), number=calcs)
#print(time_np)
mse_sklearn = round(sk.mean_squared_error(observed,predicted),5)
time_sk = it.timeit('sk.mean_squared_error(observed,predicted)', globals=globals(), number=calcs)
#print(time_sk)

for mse, mse_method, mse_time in zip([mse_vanilla, mse_numpy, mse_sklearn],
                                     ['vanilla', 'numpy', 'sklearn'],
                                     [time_sk, time_np, time_sk]):
    print(f'Mean Squared Error, {mse_method}:', mse,
          f'Execution time: ', {mse_time})

assert (mse_vanilla == mse_numpy == mse_sklearn) , "MSE methods results differ"