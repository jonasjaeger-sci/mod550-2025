# Loading the library with the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

RSEED = 77
np.random.seed(RSEED)
# Load data
iris = load_iris()
# iris.data = np.delete(iris.data,2, axis=1)
idx = [0,2,3]
print(iris.feature_names[1])
x = pd.DataFrame(iris.data, columns=iris.feature_names)
#x = x.drop("sepal width (cm)", axis=1)

# use the provided integer labels (0,1,2)
y = iris.target

# ["setosa","versicolor","virginica"]
class_names = iris.target_names

# Quick peek
print(x.head().to_string(index=False))

#"""

# Train/test split (stratified to preserve class ratios)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=RSEED)

#print(f"\nTrain size: {len(x_train)}   Test size: {len(x_test)}")

# Model
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=RSEED,
    n_jobs=-1
)

# Fit & predict
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")


# Confusion matrix with readable labels
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion matrix:")
print(cm_df)


# Cross-validated accuracy (optional but useful)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RSEED)
cv_scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"\n5-fold CV accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

# Feature importance (sorted)
fi = pd.Series(clf.feature_importances_, index=x.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(fi)
#"""