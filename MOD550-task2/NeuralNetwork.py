import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import scipy.stats

# generate linear data
dataset = DataGenerator(nRows=1000)
dataset.linear(m=1 , b=60)
dataset.add_gaussian_noise(std = 25)
x = dataset.data[:,0].reshape(-1,1)
#print(x)
y = dataset.data[:,1].reshape(-1,1)
#print(y)

# define the neural network model for linesr regression
model = Sequential([
    Dense(16,input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(16,input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(16,input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(16,input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1) # Output layer without activation for linear regression
])

# model.compile
model.compile(optimizer="adam", loss="mean_squared_error")

# train the model
model.fit(x,y,epochs=40, verbose=1)

# model prediction
y_pred = model.predict(x)
"""
if isinstance(x, np.ndarray):
    print("is an array")
elif isinstance(x, list):
    print("is a list")
"""

r_numpy = np.corrcoef(np.transpose(x), np.transpose(y))
r_scipy = scipy.stats.pearsonr(x, y_pred)
print(x.shape)
print(y_pred.shape)
print(r_numpy)
print(r_scipy)

plt.scatter(x, y, s=100, color="green", label="Experimental")
plt.plot(x, y_pred, linewidth=3, color="red", label="NN prediction")
plt.legend()
plt.grid(True)
plt.show()

# calculate MSE