import numpy as np
import  random

class DataAkquisition:
    """
    initialize data akquisition for Task2: make a 2d heat map from a 2d random distribution
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
        """
        function to construct random/uniform datapoints with lower and upper bounds (ln and ub)

        parameters:
        -----------
        lb: int
            lower bound of the generated data
        ub: int
            upper bound of the generated data

        return
        ------

        """
        self.data = np.random.uniform(lb, ub, (self.nRows, self.nCols))

    def gaussian_dist(self, mean=0, std=1):
        """
        function to construct a dataset with gaussian/normal distribution with asigned mean and standard deviation

        parameter
        ----------
        mean: float
            mean of the gaussian distribution
        std: float
            standard deviation of the gaussian distribution
        return
        ------

        """
        self.data = np.random.normal(mean,std,(self.nRows,self.nCols))