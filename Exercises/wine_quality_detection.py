"""
Assignment: Build a random forest classifier for wine quality prediction
"""

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

# define random seed
r_seed = 42

# step 1: data acquisition and preparation
filename = "/home/jonas/OneDrive/PhD/Research/MOD550/wine_quality.csv"
X = pd.read_csv(filename)
col_headers = list(X)
print(col_headers)
Y = X[col_headers[-2]]
#print(Y)
drop_col = [col_headers[i] for i in [0, 12, 13]]
X= X.drop(drop_col, axis=1)
#print(list(X))

class_names = list(set(Y))
print(class_names)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=r_seed)

#print(f"\nTrain size: {len(x_train)}   Test size: {len(x_test)}")

# Model
clf = RandomForestClassifier(n_estimators=100,
                             random_state=r_seed,
                             n_jobs=-1)

# Fit & predict
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")

# Confusion matrix with readable labels
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion matrix:")
print(cm_df)

# Cross-validated accuracy (optional but useful)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r_seed)
cv_scores = cross_val_score(clf, X, Y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"\n5-fold CV accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

# Feature importance (sorted)
fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(fi)