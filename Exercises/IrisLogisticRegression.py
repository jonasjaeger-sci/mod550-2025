import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# initiate random seed
r_seed = 77

# acquire and prepare data
iris = load_iris()
X = iris.data
Y = iris.target
#print(X.shape)
#print(Y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=r_seed)

# first solve with decision tree
DT_clf = DecisionTreeClassifier(max_leaf_nodes=10,
                                criterion="entropy",
                                random_state=r_seed)
DT_clf.fit(x_train, y_train)
plt.figure(figsize=(10,5), dpi= 60)
plot_tree(DT_clf, proportion=True, filled=True)
y_pred_DT = DT_clf.predict(x_test)
acc_DT = accuracy_score(y_test, y_pred_DT)
print(f"Accuracy of Decision Tree: {acc_DT:.3f}")

# solve with random forest
RF_clf = RandomForestClassifier(n_estimators=100,
                                random_state=r_seed,
                                n_jobs=-1)
RF_clf.fit(x_train,y_train)
y_pred_RF = RF_clf.predict(x_test)
acc_RF = accuracy_score(y_test, y_pred_RF)
print(f"Accuracy of Random Forest: {acc_RF:.3f}")

# solve with logistic regression
Lr_clf = LogisticRegression(random_state=r_seed,
                            max_iter=10,
                            solver="newton-cholesky")
Lr_clf.fit(x_train, y_train)
y_pred_LR = Lr_clf.predict(x_test)
acc_LR = accuracy_score(y_test, y_pred_LR)
print(f"Accuracy of Logistic Regression: {acc_DT:.3f}")

#plt.show()