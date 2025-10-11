"""
This file defines the DataModel class to read and process acquired data
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2

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
        self.split_bit = 0

    def linear_regression(self):
        """
        function to perform a linear regression of the dataset with library

        Returns
        -------

        """
        self.x = self.data[:,0]
        self.X = self.x.reshape(-1,1)
        self.y = self.data[:,1]

        self.lin_reg_model = LinearRegression()
        self.lin_reg_model.fit(self.X,self.y)
        self.lin_reg_y_predict = self.lin_reg_model.predict(self.X)

    def data_split(self,r_train=0.6, r_validation=0.2):
        """
        function to split the dataset into subsets for training, validation and testing
        Parameters
        ----------
        r_train: float
            ration of the data for training
        r_validation:float
            ratio nof the data for validation

        Returns
        -------

        """
        n_rows = self.data.shape[0]
        train_end_id = math.floor(n_rows * r_train)
        #print(f"train_end:{train_end_id}")
        val_end_id = math.ceil(n_rows * (r_train + r_validation))
        #print(f"val_end:{val_end_id}")
        ids = np.arange(n_rows)
        np.random.shuffle(ids)
        train_ids = ids[0 : train_end_id]
        train_ids.sort()
        val_ids = ids[train_end_id : val_end_id]
        val_ids.sort()
        test_ids = ids[val_end_id:]
        test_ids.sort()

        #print(f"train_id: {len(train_ids)}")
        #print(f"val_id: {len(val_ids)}")
        #print(f"test_id: {len(test_ids)}")

        #print(f"train_id: {train_ids}")
        #print(f"val_id: {val_ids}")
        #print(f"test_id: {test_ids}")

        self.x_train = self.data[train_ids,0:-1]
        self.y_train = self.data[train_ids,-1]
        self.x_validation = self.data[val_ids, 0:-1]
        self.y_validation = self.data[val_ids, -1]
        self.x_test = self.data[test_ids, 0:-1]
        self.y_test = self.data[test_ids, -1]

        self.split_bit = 1

    def mean_square_error(self, observation, prediction):
        """
        function to calculate the mean square error of the observed data and the model prediction
        Parameters
        ----------
        observation: float array
            observed data - truth
        prediction: float array
            predicted outcome of the model

        Returns
        -------

        """
        if len(observation) != len(prediction):
            raise ValueError(f"Unequal number of elements:\n Observation: {len(observation)} \n {len(prediction)}.")
        self.mse = sum((observation - prediction)**2) / len(observation)
        print(f"mean-square-error: {self.mse}")

    def neural_network(self,n_neurons=16, n_layers=5, act_fun="relu", l2_val=0.01,
                       opt="adam", loss_fun="mse", n_epochs=30, plot_check="no" ):
        """
        function to train a neural network that allows variable number of layers and
        choice of activation function
        Parameters
        ----------
        n_layers: int
            number of layers
        act_fun: str
            name of activation function for the hidden layer
        l2_val: float
            value for l2-regularization parameter to mitigate overfitting
        Returns
        -------

        """
        if n_layers < 1:
            print("Invalid number of layers, must be < 0. NN will be created with one layer")
        self.NN_model = Sequential()

        # setup neural network model
        # add first layer manually
        self.NN_model.add(Dense(units=n_neurons, input_dim=1,
                        activation=act_fun, kernel_regularizer=l2(l2_val)))

        # add further layers variably
        if n_layers > 1:
            for i in range(n_layers-1):
                self.NN_model.add(Dense(units=n_neurons, input_dim=1,
                                activation=act_fun, kernel_regularizer=l2(l2_val)))

        # add output layer
        self.NN_model.add(Dense(1))

        # compile the neural network model
        self.NN_model.compile(optimizer=opt, loss=loss_fun)

        # fit the model
        if self.split_bit == 0:
            raise ValueError("Data split has not been performed yet!")
        else:
            self.NN_model.fit(self.x_train, self.y_train,
                      epochs=n_epochs, verbose=1)

        # predict validation and test data
        self.NN_y_train_predict = self.NN_model.predict(self.x_train)
        self.NN_y_val_predict   = self.NN_model.predict(self.x_validation)
        self.NN_y_test_predict  = self.NN_model.predict(self.x_test)

        self.NN_Rvalue_train = np.corrcoef(np.transpose(self.x_train), np.transpose(self.NN_y_train_predict))
        self.NN_Rvalue_train = np.round(self.NN_Rvalue_train,3)
        self.NN_Rvalue_val   = np.corrcoef(np.transpose(self.x_validation), np.transpose(self.NN_y_val_predict))
        self.NN_Rvalue_val = np.round(self.NN_Rvalue_val, 3)
        #if self.NN_Rvalue_val[0, 1] < 0.7:
        #    raise ValueError(f"Insufficient model performance. R-Value < 0.7 - please adapt model parameters")
        self.NN_Rvalue_test  = np.corrcoef(np.transpose(self.x_test), np.transpose(self.NN_y_test_predict))
        self.NN_Rvalue_test = np.round(self.NN_Rvalue_test, 3)

        if plot_check == "yes":
            NN_fig = plt.figure()

            # training
            ax1 = NN_fig.add_subplot(1,3,1)
            ax1.set_title(f"Neural Network performance on training data: R={self.NN_Rvalue_train[0,1]}")
            ax1.set_xlabel("x", fontsize=14, fontweight="bold")
            ax1.set_ylabel("y", fontsize=14, fontweight="bold")
            ax1.scatter(self.x_train, self.y_train, s=50,
                        color="black", label="training data")
            ax1.plot(self.x_train, self.NN_y_train_predict, color="red",
                     linewidth=3, linestyle ="--", label="prediction")
            ax1.legend()
            ax1.grid(True)

            # validation
            ax1 = NN_fig.add_subplot(1, 3, 2)
            ax1.set_title(f"Neural Network performance on validation data: R={self.NN_Rvalue_val[0, 1]}")
            ax1.set_xlabel("x", fontsize=14, fontweight="bold")
            ax1.set_ylabel("y", fontsize=14, fontweight="bold")
            ax1.scatter(self.x_validation, self.y_validation, s=50,
                        color="black", label="validation data")
            ax1.plot(self.x_validation, self.NN_y_val_predict, color="blue",
                     linewidth=3, linestyle="--", label="prediction")
            ax1.legend()
            ax1.grid(True)

            # test
            ax1 = NN_fig.add_subplot(1, 3, 3)
            ax1.set_title(f"Neural Network performance on test data: R={self.NN_Rvalue_test[0, 1]}")
            ax1.set_xlabel("x", fontsize=14, fontweight="bold")
            ax1.set_ylabel("y", fontsize=14, fontweight="bold")
            ax1.scatter(self.x_test, self.y_test, s=50,
                        color="black", label="test data")
            ax1.plot(self.x_test, self.NN_y_test_predict, color="green",
                     linewidth=3, linestyle="--", label="prediction")
            ax1.legend()
            ax1.grid(True)

    def k_means(self, max_n_clusters=5):
        """
        function to train a k-means model and divide the unlabeled data into clusters
        Parameters
        ----------
        max_n_clusters: int
            maximum number of clusters to test

        Returns
        -------

        """

        # run multiple clusters to see which one is efficient enough
        sum_squared_dist = []
        for i in range(1,max_n_clusters+1):
           km = KMeans(n_clusters=i)
           km.fit(self.data[:,-1].reshape(-1,1))
           sum_squared_dist.append(km.inertia_)

        fig_km1 = plt.figure()
        ax = fig_km1.add_subplot(1,1,1)
        ax.set_title(f"Evaluation of sum of squared distances \n"
                     f"for different clusters")
        ax.set_xlabel("# of clusters", fontsize=15, fontweight="bold")
        ax.set_ylabel("# sum of squared distances", fontsize=15, fontweight="bold")
        ax.plot(range(1, max_n_clusters+1), sum_squared_dist, color="black",
                linewidth=3, marker="x")
        ax.grid(True)
        plt.show()

        # evaluate the plot and decide how many clusters are sufficient
        n_cluster = int(input("Evaluate plot and insert number of clusters to fit the model: "))

        # train model and display the clusters
        self.km_model = KMeans(n_clusters=n_cluster)
        self.km_model.fit(self.data[:,-1].reshape(-1,1))
        cluster_labels = self.km_model.labels_

        fig_km2 = plt.figure()
        ax = fig_km2.add_subplot(1,1,1)
        ax.set_title(f"K-means clustering with {n_cluster} clusters")
        ax.set_xlabel("x", fontsize=15, fontweight="bold")
        ax.set_ylabel("y", fontsize=15, fontweight="bold")
        ax.scatter(self.data[:,0],self.data[:,1], c=cluster_labels, s=50)
        ax.grid(True)

    def gmm(self, max_n_clusters=5):
        """
        function to define a gaussian mixture model (gmm) and divide
        the unlabeled data into clusters which each represent a gaussian distribution
        Parameters
        ----------
        max_n_clusters: int
            maximum number of clusters to test

        Returns
        -------

        """

        # run multiple clusters to see which one is efficient enough
        bic = [] # bayesian information criterion, comparable to inertia from kmeans
        for i in range(1,max_n_clusters+1):
            gmm = GaussianMixture(n_components=i)
            gmm.fit(self.data[:,-1].reshape(-1,1))
            bic.append(gmm.bic(self.data[:,-1].reshape(-1,1)))

        fig_gmm1 = plt.figure()
        ax = fig_gmm1.add_subplot(1,1,1)
        ax.set_title(f"Evaluation of bayesian information criterion \n"
                     f"for different clusters")
        ax.set_xlabel("# of clusters", fontsize=15, fontweight="bold")
        ax.set_ylabel("# Bayesian information criterion", fontsize=15, fontweight="bold")
        ax.plot(range(1, max_n_clusters+1), bic, color="black",
                linewidth=3, marker="x")
        ax.grid(True)
        plt.show()

        # evaluate the plot and decide how many clusters are sufficient
        n_cluster = int(input("Evaluate plot and insert number of clusters to fit the model: "))

        # train model and display the clusters
        self.gmm_model = GaussianMixture(n_components=n_cluster)
        self.gmm_model.fit(self.data[:, -1].reshape(-1, 1))
        cluster_labels = self.gmm_model.predict(self.data[:, -1].reshape(-1, 1))

        print(f"cluster labels:{cluster_labels}")

        fig_gmm2 = plt.figure()
        ax = fig_gmm2.add_subplot(1, 1, 1)
        ax.set_title(f"gaussian mixture model clustering with {n_cluster} clusters")
        ax.set_xlabel("x", fontsize=15, fontweight="bold")
        ax.set_ylabel("y", fontsize=15, fontweight="bold")
        ax.scatter(self.data[:, 0], self.data[:, 1], c=cluster_labels, s=50)
        ax.grid(True)

