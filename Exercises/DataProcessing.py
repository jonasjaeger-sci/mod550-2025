import numpy as np
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator

test = DataGenerator()
test.normal_dist()
plt.scatter(*test.data)
plt.show()