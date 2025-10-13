"""
script that defines the DataGenerator class to create different datasets
that can be manipulated and analysed
"""

import numpy as np

class DataGenerator:
    """
    initialize class DataGenerator to construct various datasets
    """
    def __init__(self, n_rows=1000, n_cols=2):
        """
        Initialization function that describes the size of the dataset
        Parameters
        ----------
        n_rows: int
            number of experiments
        n_cols: int
            number of features

        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.data = []

    def uniform_dist(self, lb=0., ub=1.):
        """
        function to generate data with uniform distribution
        Parameters
        ----------
        lb: float
            lower boundary
        ub: float
            upper boundary

        """
        self.data = np.random.uniform(lb,ub,(self.n_rows,self.n_cols))

    def gaussian_dist(self, mean=0., std=1.):
        """
        function to generate data with gaussian distribution around given
        mean and standard deviation
        Parameters
        ----------
        mean: float
            mean of the distribution
        std: float
            standard deviation of the distribution

        """
        self.data = np.random.normal(mean,std,(self.n_rows,self.n_cols))

    def linear(self,x_0=0., x_end=1000., m=1., b=0.):
        """
        function to generate linear data
        Parameters
        ----------
        x_0: float
            start value for x
        x_end: float
            end value for x
        m: float
            slope of the data
        b: float

        """
        x = np.linspace(x_0, x_end, self.n_rows)
        y = m*x + b
        self.data = np.column_stack((x,y))

    def quadratic(self,x_0=0., x_end=1000., a=1., b=0., c=0.):
        """
        function to generate quadratic data
        Parameters
        ----------
        x_0: float
            start value for x
        x_end: float
            end value for x
        a: float
            coefficient
        b: float
            coefficient
        c: float
            coefficient

        """
        x = np.linspace(x_0, x_end, self.n_rows)
        y = a*x**2 + b*x + c
        self.data = np.column_stack((x,y))

    def sine(self, A=1., k=1., p_shift=0., C=0.):
        """

        Parameters
        ----------
        A: float
            Amplitude
        k: float
            angular frequency
        p_shift: float
            phase shift
        C: float
            vertical shift
        """
        x = np.linspace(0,2*np.pi,self.n_rows)
        y = A * np.sin(k*x + p_shift) + C
        self.data = np.column_stack((x,y))

    def exponential(self, ub=10, k=1):
        """
        function to generate exponential data
        Parameters
        ----------
        ub: float
            upper bound of the x-sequence
        k: float
            exponential coefficient

        """
        x = np.linspace(0,ub,self.n_rows)
        y = np.exp(k*x)
        self.data = np.column_stack((x,y))

    def add_gaussian_noise(self, col=1, mean=0., std=1.):
        """
        function to add gaussian distributed noise to existing data
        around a given mean and standard deviation
        Parameters
        ----------
        col: int
            column where the noise shall be added
        mean: float
            mean of the distribution
        std: float
            standard deviation of the distribution

        """
        self.data[:,col] += np.random.normal(mean,std,self.n_rows)

    def add_uniform_noise(self, col=1, lb=0., ub=1.):
        """
        function to add random noise with given upper and lower boundaries
        to existing data
        Parameters
        ----------
        col: int
            column where the noise shall be added
        lb: float
            lower boundary
        ub: float
            upper boundary

        """
        self.data[:,col] += np.random.uniform(lb,ub,self.n_rows)
