import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)
plt.figure(figsize=(10,5), dpi= 60)
clf = DecisionTreeClassifier(max_leaf_nodes=10, criterion="entropy")

clf.fit(X, y)
plot_tree(clf, proportion=True, filled=True)

plt.show()