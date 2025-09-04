import numpy as np
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator

te = DataGenerator()
#te.uniform_dist()
#te.gaussian_dist()
#plt.hist(te.data[:,1])
#te.linear(m=0.6,b = 10)
#te.quadratic()
#te.sine()
te.exponential(k = -1)
te.add_gaussian_noise(std = 0.2)
#te.add_uniform_noise(ub = 200)
#print(te.data.shape)
plt.scatter(*te.data.T)

plt.show()