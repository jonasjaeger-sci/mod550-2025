
from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit as it
import numpy as np

np.random.seed(0)
n = 10
predicted = np.linspace(-10,10,n)
observed = predicted + np.random.normal(0,1.5,n)

kwarg = {'obs': observed, 'pred': predicted}

functions= {'mse_vanilla': vanilla_mse,
            'mse_numpy': numpy_mse,
            'mse_sk': sk_mse}

#calcuate MSE
calcs = 1000

index = functions.items()
#print(kwarg.values())
#print(item[0])
#print(vanilla_mse(*kwarg))
#print(item[0](kwarg))

#"""
for method_name,method in functions.items():
    #print(method_name)
    #print(method)
    mse = round(method(*kwarg.values()),4)
    #print(mse)
    t_exec = it.timeit('{method(*kwarg.values())}', globals=globals(), number=calcs)

    print(f'Mean squared error with {method_name}:', mse,
          f'with cumulated execution time: {t_exec} seconds')
    
#"""
