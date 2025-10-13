"""
This script utilizes the DataGenerator and DataModel classes to generate,
manipulate and analyzes data with different ML approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from data_model import  DataModel

# initialize data model object
dataset = DataGenerator(n_rows=1001)
dataset.linear(x_0=-20, x_end=100, m=2.43 , b=6.1)
dataset.add_gaussian_noise(std = 25)
datmod = DataModel(dataset.data)

# Task 2.7: Make a linear regression on all data
datmod.linear_regression_vanilla(train=False)
plt.show()
