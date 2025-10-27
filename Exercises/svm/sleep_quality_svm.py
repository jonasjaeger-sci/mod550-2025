"""
Assignment: Build a support vector machine classifier for the quality of sleep
"""

import sys
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, balanced_accuracy_score,
                             precision_score)

# define random seed
r_seed = 77

# acquire and prepare data
file = ("/home/jonas/PycharmProjects/mod550-2025/ML_project_data"
        "/Sleep_health_and_lifestyle_dataset.csv")
X = pd.read_csv(file)
X[["BP_systolic", "BP_diastolic"]] = X["Blood Pressure"].str.split("/", expand=True)
X[["BP_systolic", "BP_diastolic"]] = X[["BP_systolic", "BP_diastolic"]].astype(float)
X = X.fillna(value="No")
headers = list(X)
Y = X[headers[5]]
drop_headers = [headers[i] for i in [0,5,9]]
X = X.drop(drop_headers, axis=1)
X_encoded = pd.get_dummies(X)
X_encoded = X_encoded.to_numpy()
enc_headers = list(X_encoded)

# check how balanced data is
class_names = list(set(Y))
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

# scale/normalize data
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)
X_sc = scaler.transform(X_encoded)
#sys.exit()

# Prepare a list of kernels and their parameters for iteration
kernels = [('linear', svm.SVC(kernel='linear',  random_state=r_seed)),
           ('poly (d=3)', svm.SVC(kernel='poly', degree=3, random_state=r_seed)),
           ('rbf', svm.SVC(kernel='rbf', random_state=r_seed)),
           ('sigmoid', svm.SVC(kernel='sigmoid', random_state=r_seed))]

for kernel_name, svm_clf in kernels:
    print(f"SVM: training, prediction and analysis with {kernel_name} kernel")
    svm_clf.fit(x_train_sc,y_train)
    y_test_predict = svm_clf.predict(x_test_sc)
    
    # analyze predictive performance
    acc = accuracy_score(y_test, y_test_predict)
    balanced_acc = balanced_accuracy_score(y_test, y_test_predict)
    sensitivity = recall_score(y_test, y_test_predict, average="micro")  # true positive rate
    precision = precision_score(y_test, y_test_predict, average="micro")
    f1 = f1_score(y_test, y_test_predict, average="micro")

    print(f"Accuracy: {acc:.3f}")
    print(f"balanced Accuracy: {balanced_acc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F1-Score: {f1:.3f} \n")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(f"SVM predictions with {kernel_name} kernel \n"
                 f"Accuracy = {acc}")
    ax.set_xlabel("# of sample", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sleep Quality", fontsize=14, fontweight="bold")
    ax.scatter(range(1,len(y_test)+1), y_test, color="green", marker="s",
               s=30, label="test data", alpha=0.5)
    ax.scatter(range(1,len(y_test)+1),y_test_predict, color="red", marker="o",
               s=30, label="prediction", alpha=0.5)
    ax.legend()
    ax.grid(True)

plt.show()

