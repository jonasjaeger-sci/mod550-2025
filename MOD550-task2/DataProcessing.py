"""
This code is used to utilize the predefined classes DataGenerator and DataModel to perform the tasks
"""


import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from DataModel import DataModel

# Task 1: test data acquisition model
dataset = DataGenerator(n_rows=1001)
dataset.linear(x_0=-200, x_end=dataset.n_rows, m=1 , b=60)
dataset.add_gaussian_noise(std = 25)
#plt.scatter(dataset.data[:,0],dataset.data[:,1])
#plt.show()

# Task 2: test linear regression
datamodel = DataModel(dataset.data)
datamodel.linear_regression()
#plt.scatter(datamodel.x,datamodel.y, label="")
#plt.plot(datamodel.x,datamodel.lin_reg_y_predict,color="red", linewidth=3)

# Task 3: test data split function
datamodel.data_split()
#plt.scatter(datamodel.x_train,datamodel.y_train,color="red")
#plt.scatter(datamodel.x_validation,datamodel.y_validation,color="blue")
#plt.scatter(datamodel.x_test,datamodel.y_test,color="green")

# Task 4: test MSE function
datamodel.mean_square_error(observation=datamodel.y, prediction=datamodel.lin_reg_y_predict)

#"""
#Task 5: test NN
#datamodel.neural_network(act_fun="squareplus", plot_check="yes")
#y_val_predict = datamodel.NN_model.predict(datamodel.x_validation)
#plt.scatter(datamodel.x, datamodel.y, color="black",s=20, label="all")
#plt.scatter(datamodel.x_validation, datamodel.y_validation, color="blue",s=20, label="val")
#plt.plot(datamodel.x_validation, y_val_predict, color="red", linewidth=2, label="val_predict")
#plt.legend()
#print(datamodel.x_validation)
#plt.show()
#"""

# Task 6: test kmeans and gaussian mixture model (gmm)
#datamodel.k_means(max_n_clusters=11)
#plt.show()

datamodel.gmm(max_n_clusters=11)
plt.show()