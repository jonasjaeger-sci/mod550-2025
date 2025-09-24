"""
This file defines the DataModel class to read and process acquired data
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class DataModel:

    def __init__(self,data):
        """
        Initialization of DataModel instance

        Parameters
        ----------
        data: numpy array or pandas dataframe
            dataset passed along from the DataAcquisition instance
        """
        self.data = data

    def linear_regression(self):
        """
        function to perform a linear regression of the dataset

        Returns
        -------

        """

        self.x = self.data[:,0]
        self.X = self.x.reshape(-1,1)
        self.y = self.data[:,1]

        self.lin_reg_model = LinearRegression()
        self.lin_reg_model.fit(self.X,self.y)
        self.lin_reg_y_predict = self.lin_reg_model.predict(self.X)
