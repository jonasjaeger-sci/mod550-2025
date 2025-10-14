"""
Assignment: Build a logistic regression classifier for the quality of sleep
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, balanced_accuracy_score,
                             precision_score)

# define random seed
r_seed = 77

# acquire and prepare data
file = "/home/jonas/PycharmProjects/mod550-2025/ML_project_data/Sleep_health_and_lifestyle_dataset.csv"
X = pd.read_csv(file)
X[["BP_systolic", "BP_diastolic"]] = X["Blood Pressure"].str.split("/", expand=True)
X[["BP_systolic", "BP_diastolic"]] = X[["BP_systolic", "BP_diastolic"]].astype(float)
X = X.fillna(value="No")
headers = list(X)
#print(headers)
Y = X[headers[5]]
drop_headers = [headers[i] for i in [0,5,9]]
# print(drop_headers)
X = X.drop(drop_headers, axis=1)
print(list(X))
X_encoded = pd.get_dummies(X)
print(list(X_encoded))
enc_headers = list(X_encoded)


class_names = list(set(Y))
#print(class_names)
occurences = {}
occ_weights = {}
for str in (class_names):
    occurences[str] = sum(Y == str)
    occ_weights[str] = round(occurences[str]/len(Y),3)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.set_xlabel("Classes", fontsize=14, fontweight="bold")
ax1.set_ylabel("Occurence ratio", fontsize=14, fontweight="bold")
ax1.bar(occ_weights.keys(), occ_weights.values(), color="red", edgecolor="black", linewidth=1)
ax1.grid(True)

# split data, train model and predict outcome
x_train, x_test, y_train, y_test = train_test_split(
    X_encoded, Y, test_size=0.25, random_state=r_seed)

Lr_clf = LogisticRegression(random_state=r_seed,
                            max_iter=100,
                            solver="newton-cholesky")

#"""
Lr_clf.fit(x_train, y_train)
y_pred = Lr_clf.predict(x_test)

# prediction analysis
# Metrics
acc = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, average="micro") # true positive rate
precision = precision_score(y_test, y_pred, average="micro")
f1 = f1_score(y_test, y_pred, average="micro")

print(f"\nAccuracy: {acc:.3f}")
print(f"\nbalanced Accuracy: {balanced_acc:.3f}")
print(f"\nSensitivity: {sensitivity:.3f}")
print(f"\nPrecision: {precision:.3f}")
print(f"\nF1-Score: {f1:.3f}")

#plt.show()
#"""
