"""
Assignment: Build a logistic regression classifier for wine quality prediction
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             multilabel_confusion_matrix, recall_score, balanced_accuracy_score,
                             precision_score)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define random seed
r_seed = 77

# step 1: data acquisition and preparation
filename = "/home/jonas/OneDrive/PhD/Research/MOD550/wine_quality.csv"
X = pd.read_csv(filename)
col_headers = list(X)
#print(col_headers)
Y = X[col_headers[-2]]
#print(len(Y))
drop_col = [col_headers[i] for i in [0, 12, 13]]
X= X.drop(drop_col, axis=1)
#print(list(X))

class_names = list(set(Y))
#print(class_names)
occurences = {}
occ_weights = {}
for str in (class_names):
    occurences[str] = sum(Y == str)
    occ_weights[str] = round(occurences[str]/len(Y),3)
#print(occurences)
#print(occ_weights)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.set_xlabel("Classes", fontsize=14, fontweight="bold")
ax1.set_ylabel("Occurence ratio", fontsize=14, fontweight="bold")
ax1.bar(occ_weights.keys(), occ_weights.values(), color="red", edgecolor="black", linewidth=1)
ax1.grid(True)


# split data, train model and predict outcome
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=r_seed)

#print(f"\nTrain size: {len(x_train)}   Test size: {len(x_test)}")

# Model
Lr_clf = LogisticRegression(random_state=r_seed,
                            max_iter=10000,
                            solver="newton-cholesky")

# Fit & predict
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

# Confusion matrix with readable labels
cm = confusion_matrix(y_test, y_pred)
mlcm = multilabel_confusion_matrix(y_test, y_pred, labels=class_names)
# output mlcm = [TN , FP
#                FN , TP]
#print(cm)
#print(f"multilabel confusion matrix: \n {mlcm}")
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
#print("\nConfusion matrix:")
#print(cm_df)

# Cross-validated accuracy (optional but useful)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r_seed)
cv_scores = cross_val_score(Lr_clf, X, Y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"\n5-fold CV accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

#plt.show()