import numpy as np
import pandas as pd

class DataAkquisition:
    """
    initialize data akquisition for Task3: make a histogram for the source data you selected
    """
    def __init__(self):
        pass

    def read_csv(self,path,delimiter,headers=[]):
        """
        function to read csv file into accessible data
        parameter
        ---------
        path: str
            string representing the path of the .csv file
        delimiter: str
            string to be applied for the data separation
        headers: str list
            list of strings containing the headers for the data
        return
        ------
        pandas dataframe
        """
        data = pd.read_csv(path,sep=delimiter,decimal=",")

        if headers:
            data.columns = headers

        return data