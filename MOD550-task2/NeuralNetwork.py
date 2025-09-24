import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from DataGenerator import DataGenerator

# generate linear data
dataset = DataGenerator()
dataset.linear(m=1 , b=60)
dataset.add_gaussian_noise(std = 25)
x = dataset.data[:,0].reshape(-1,1)
y = dataset.data[:,1].reshape(-1,1)

# define the neural network model for linesr regression
model = Sequential([
    Dense(8,input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(64,input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1) # Output layer without activation for linear regression
])

# model.compile

# model.fit






