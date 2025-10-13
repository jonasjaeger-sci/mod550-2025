import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:
    """
    initialize class DataGenerator to construct various datasets
    """
    def __init__(self, n_rows=1000, n_cols=2):
        """
        Initialization function that describes the size of the dataset
        :param n_rows:
        :param n_cols:
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.data = []

    def uniform_dist(self, lb=0, ub=1):
        #self.data = np.random.rand(self.n_rows,self.n_cols)
        self.data = np.random.uniform(lb,ub,(self.n_rows,self.n_cols))

    def gaussian_dist(self, mean=0, std=1):
        self.data = np.random.normal(mean,std,(self.n_rows,self.n_cols))

    def linear(self, m=1, b=0):
        x = np.arange(0,self.n_rows,1)
        y = m*x + b
        self.data = np.column_stack((x,y))

    def quadratic(self, a=1, b=0, c=0):
        x = np.arange(0,self.n_rows,1)
        y = a*x**2 + b*x + c
        self.data = np.column_stack((x,y))

    def sine(self, A=1, k=1, p_shift=0, C=0):
        x = np.linspace(0,2*np.pi,self.n_rows)
        y = A * np.sin(k*x + p_shift) + C
        self.data = np.column_stack((x,y))

    def exponential(self, ub=10, k=1):
        x = np.linspace(0,ub,self.n_rows)
        y = np.exp(k*x)
        self.data = np.column_stack((x,y))

    def add_gaussian_noise(self, mean=0, std=1):
        self.data[:,1] += np.random.normal(mean,std,self.n_rows)

    def add_uniform_noise(self, lb=0, ub=1):
        self.data[:,1] += np.random.uniform(lb,ub,self.n_rows)



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


