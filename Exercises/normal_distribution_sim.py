import numpy as np
import matplotlib.pyplot as plt

mu = 5
std = 4

dist = np.random.normal(mu,std,1000)
min = np.min(dist)
max = np.max(dist)
n_bins = 20
delta = (max-min)/n_bins
bins = np.arange(min,max+delta,delta)
plt.hist(dist,bins=bins)
plt.show()

