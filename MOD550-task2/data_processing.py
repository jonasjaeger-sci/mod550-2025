"""
This script utilizes the DataGenerator and DataModel classes to generate,
manipulate and analyzes data with different ML approaches
"""

import matplotlib.pyplot as plt
from data_generator import DataGenerator
from data_model import  DataModel

# initialize data model object
dataset = DataGenerator(n_rows=1001)
dataset.linear(x_0=-20, x_end=100, m=2.43 , b=6.1)
#dataset.sine()
dataset.add_gaussian_noise(mean=5.1, std = 25)
datmod = DataModel(dataset.data)

# Task 2.7: Make a linear regression on all data
datmod.linear_regression_vanilla(train=False)

# Task 2.8: Make a linear regression on train data and test it on validation set
datmod.data_split()
datmod.linear_regression_vanilla(train=True)

test = datmod.x_validation.ravel()
datmod.y_val_predict = datmod.m * datmod.x_validation + datmod.b

fig_LR_val = plt.figure()
ax = fig_LR_val.add_subplot(1,1,1)
ax.set_xlabel("x", fontsize=14, fontweight="bold")
ax.set_ylabel("y", fontsize=14, fontweight="bold")
ax.scatter(datmod.x_validation, datmod.y_validation, color="black", s= 50, label="data points")
ax.plot(datmod.x_validation, datmod.y_val_predict, linewidth=2.5,
        color="red", label="linear regression")
ax.legend()
ax.grid(True)

# Task 2.9: Compute MSE for validation data
datmod.mean_square_error(observation=datmod.y_validation,
                         prediction=datmod.y_val_predict)

# Task 2.10
# a) Discuss how different functions can be used in the linear regression
# and different NN architecture:
"""
For the linear regression we can apply a more general model description.
The general form of the linear model function is q(x) = sum_i^n b_i * f_i(x). This means we 
can construct the model as a linear combination of different and even non-linear functions
to describe more complex lines. For this multi-variable equation we can try to find
the coefficients with an analytical solution. However, it would be more efficient to calculate
the coefficients with matrix algebra in a multi linear regression: B=(X.T * X)^-1 * X.T * Y. To 
see which model better describes the observations, it is furthermore important to utilize loss
functions like the mean square error or the entropy.
"""

datmod.neural_network(act_fun="relu", loss_fun="mse", plot_check=True)
datmod.neural_network(act_fun="linear", loss_fun="mse", plot_check=True)

"""
Beside different loss functions, the training and predictive capabilities of a neural network 
rely furthermore on which activation functions are used. The activation functions describe how
the nodes themselves process the input and impact the output. 
"""

# b) Discuss how you can use the validation data for the different cases
"""
The validation data as a subset of the real dataset is used as a first indicator of how well 
the model performs after training i.e. coefficient determination. Calculating the loss from 
the validation predictions can be used to further optimize some of the model parameters. It 
should be noted, that for unbalanced datasets a special emphasis should be placed on assuring 
that ever subset contains datapoints with the same ratio
"""

# c)Discuss the different outcome from the different models when using the full dataset to train
# and when you use a different ML approach.
"""
Using all datapoints to train the ML model can prove problematic since it is not possible
to tell how the model performs on unseen data. The model could therefore be very precise
on predicting the datapoints used for training but totally go off when trying to predict new data,
indicating overfitting of the model.

The ML approach should be chosen in accordance to the data and also the size. For example,
if I have a lot of unlabeled data, it might be useful to apply unsupervised learning
methods to generate meaningful labels such as k-means or PCA. If labeled target data is available,
it might be more sensible to incorporate a supervised learning approach like decision tree 
or random forest.
"""

# d) Discuss the outcomes you get for K-means and GMM
datmod.k_means(max_n_clusters=11)
datmod.gmm(max_n_clusters=11)

"""
For k-means and gmm it is necessary to decide how many clusters should be created. In the case of 
k-means I calculated the sum of squared distances and plot them with the number of corresponding 
clusters. The most efficient choice is where the graph shows an inflection point, in this case 
four clusters.
As for gmm, the number of clusters is determined here with the bayesian information criterion (bic).
The optimal number of clusters is where the bic is at a minimum, in this case 3. 
"""

# e) Discuss how you can integrate supervised and unsupervised methods for your case
"""
In the case of the prediction of market prices for electricity it both supervised and unsupervised 
approaches can be utilized. As for supervised methods it might be useful to discretize the
market prizes granularly and use a decision tree or random forest to predict the prices 
depending on features like intensity of sun, wind and installed capacities. Furthermore, it might
be useful to incorporate k-means to find hidden clusters for example in the production of 
wind energy in dependence of wind speed or time of day.
"""

plt.show()
