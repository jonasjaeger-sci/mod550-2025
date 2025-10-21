import numpy as np

n_samples= 10000

# 3 estimate integral 0-1 exp(-x²)
exp= []

for i in range(n_samples):
    x = np.random.uniform(0,1.00001)
    exp.append(np.exp(-x**2))

mc_exp = sum(exp)/n_samples
print(f"mc_exp:\n{mc_exp}")