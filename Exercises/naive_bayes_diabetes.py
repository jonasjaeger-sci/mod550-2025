"""
Assignment: Build a support vector classifier for wine quality prediction
"""

import sys
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, balanced_accuracy_score,
                             precision_score)

# define random seed
r_seed = 77

# data acquisition and preparation
filename = ("/home/jonas/PycharmProjects/mod550-2025/ML_project_data/"
            "diabetes_risk_bernoulli_full.csv")
X = pd.read_csv(filename)
print(X)
col_headers = list(X)
Y = X[col_headers[-1]]
print(Y)


# split data, train model and predict outcome
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=r_seed)

BNB = BernoulliNB(alpha=1.0)


