"""
This code is used to utilize the predefined classes DataGenerator and DataModel to perform the tasks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from DataModel import DataModel

dataset = DataGenerator()
dataset.linear(m=1 , b=60)
dataset.add_gaussian_noise(std = 25)

#plt.scatter(dataset.data[:,0],dataset.data[:,1])
#plt.show()

#"""
datamodel = DataModel(dataset.data)
datamodel.linear_regression()
plt.scatter(datamodel.x,datamodel.y)
plt.plot(datamodel.x,datamodel.lin_reg_y_predict,color="red", linewidth=3)
plt.show()
#"""
