import numpy as np
import random
import matplotlib.pyplot as plt

class DataGenerator:
    """
    initialize class DataGenerator to construct various datasets
    """
    def __init__(self, nRows=1000, nCols=2):
        """
        Initialization function that describes the size of the dataset
        :param nRows:
        :param nCols:
        """
        self.nRows = nRows
        self.nCols = nCols
        self.data = []

    def uniform_dist(self, lb=0, ub=1):
        #self.data = np.random.rand(self.nRows,self.nCols)
        self.data = np.random.uniform(lb,ub,(self.nRows,self.nCols))

    def gaussian_dist(self, mean=0, std=1):
        self.data = np.random.normal(mean,std,(self.nRows,self.nCols))

    def linear(self, m=1, b=0):
        x = np.arange(0,self.nRows,1)
        y = m*x + b
        self.data = np.column_stack((x,y))

    def quadratic(self, a=1, b=0, c=0):
        x = np.arange(0,self.nRows,1)
        y = a*x**2 + b*x + c
        self.data = np.column_stack((x,y))

    def sine(self, A=1, k=1, p_shift=0, C=0):
        x = np.linspace(0,2*np.pi,self.nRows)
        y = A * np.sin(k*x + p_shift) + C
        self.data = np.column_stack((x,y))

    def exponential(self, ub=10, k=1):
        x = np.linspace(0,ub,self.nRows)
        y = np.exp(k*x)
        self.data = np.column_stack((x,y))

    def add_gaussian_noise(self, mean=0, std=1):
        self.data[:,1] += np.random.normal(mean,std,self.nRows)

    def add_uniform_noise(self, lb=0, ub=1):
        self.data[:,1] += np.random.uniform(lb,ub,self.nRows)



"""
te = DataGenerator()
#te.uniform_dist()
#te.gaussian_dist()
#plt.hist(te.data[:,1])
#te.linear(m=0.6,b = 10)
#te.quadratic()
te.sine()
te.exponential(k = -1)
#te.add_gaussian_noise(std = 0.2)
#te.add_uniform_noise(ub = 200)
#print(te.data.shape)
plt.scatter(*te.data.T)

plt.show()
"""


