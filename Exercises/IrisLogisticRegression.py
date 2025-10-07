import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# first solve with decision tree
iris = load_iris()
X = iris.data
y = iris.target
#print(X.shape)
#print(y.shape)
plt.figure(figsize=(10,5), dpi= 60)
clf = DecisionTreeClassifier(max_leaf_nodes=10, criterion="entropy")

clf.fit(X, y)
plot_tree(clf, proportion=True, filled=True)

# solve with logistic regression
Lr = LogisticRegression()
Lr.fit(X,y)
Lr.score(X,y)

plt.show()