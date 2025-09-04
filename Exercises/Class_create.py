import numpy as np
import random
import matplotlib.pyplot as plt


random.seed(1)

class TestClass:
    """
    Demo class
    """
    def __init__(self,name):
        """
        Initialization of class object

        :param
        ------
        name: string
            name of the instance created
        """
        self.name = name

    def scream(self,sound):
        print(self.name + " screams in agony: " + sound)

Instance1 = TestClass("Josef")
Instance1.scream("Ahhhhhhhhhh!!!")

class MyFirstClass:
    """
    Demo Class
    """
    def __init__(self,input):
        """
        Initialize class instance

        :param self:
        :param input: float
            A number
        """
        # print(input)

    def make_list(self,start=1,end=101,numEl=10):
        """
        function to create a list in ascending order

        :param start: int
            start of list
        :param end: int
            end of list
        :param numEl: int
            number of list elements
        :return regular_list
            a list of ascending numbers
        """
        #self.regular_list = list(range(start,end))
        self.regular_list = np.round(np.linspace(start,end,numEl),2)
        self.regular_list = self.regular_list.tolist()
        # print(self.regular_list)

        return self.regular_list

    def make_rnd_list(self,start=1,end=101,numEl=10):
        """

        :param self:
        :param numEl: int
            number of list elements
        :param start: int
            start of the interval of random numbers
        :param end: int
            end of interval of random numbers
        :return: self.rnd_list
        """
        self.rnd_list = random.choices(range(start,end),k=numEl)
        self.rnd_list.sort(reverse = True)
        # print(self.rnd_list)

        return self.rnd_list

    def make_2D_list(self,start=1,end=101,numLists = 10):
        """
        function to create a 2D-list. Each entry is a list with two random elements within a certain interval
        :param start: int
            start of the interval
        :param end: int
            end of the interval
        :param numLists: int
            number of lists
        :return: self.list_2D
        """
        self.list_2D = []
        for i in range(numLists):
            self.list_2D.append(random.choices(range(start,end),k=2))
        print(self.list_2D)

        return self.list_2D

    def make_2D_array(self,nRows=100):
        """
        function to construct a nRows x 2 dimensional array with continuous numbers between 0 and 1
        :param
        ------
        nRows: int
            number of rows/instances
        :return:
        --------
        rand_2D_array: float array
            array containing random numbers
        """

        rand_2D_array = np.random.rand(nRows,2)

        return rand_2D_array

    def make_nD_array(self,nRows=100,nCols=3):
        """
        function to construct a nRows x 2 dimensional array with continuous numbers between 0 and 1
        :param
        ------
        nRows: int
            number of rows/instances
        nCols: int
            number of rows/instances
        :return:
        --------
        rand_nD_array: float array
        """

        rand_nD_array = np.random.rand(nRows,nCols)

        return rand_nD_array



Instance2 = MyFirstClass(3)
Instance2.make_list(1,101)
Instance2.make_rnd_list(1,101,10)
#test_2D_list = Instance2.make_2D_list()
#print(test_2D_list[0])

test_rand_2D_array = Instance2.make_2D_array(nRows=1000)
test_rand_nD_array = Instance2.make_nD_array(nRows = 1000,nCols = 3)

"""
fig,ax = plt.subplots(1,2)

ax[0].scatter(test_rand_2D_array[:,0],test_rand_2D_array[:,1])
ax[0].set_xlabel("random x")
ax[0].set_ylabel("random y")
ax[0].grid(True)

ax[1] = fig.add_subplot(projection="3d")
ax[1].scatter(test_rand_nD_array[:,0],test_rand_nD_array[:,1],test_rand_nD_array[:,2])

plt.show()
"""

fig2 = plt.figure()

#first plot in 2D
ax = fig2.add_subplot(1,2,1)
ax.scatter(test_rand_2D_array[:,0],test_rand_2D_array[:,1])
ax.set_xlabel("random x")
ax.set_ylabel("random y")
ax.grid(True)

# second plot in 3D
ax = fig2.add_subplot(1,2,2,projection="3d")
ax.scatter(test_rand_nD_array[:,0],test_rand_nD_array[:,1],test_rand_nD_array[:,2])

plt.show()



