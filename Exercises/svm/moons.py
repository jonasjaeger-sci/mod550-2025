

import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from pltsvm import plot_svm_results




# ========================================================================
# --- SEQUENCE 3: NON-LINEAR DATA (moons)
# ========================================================================

X_moons, y_moons = make_moons(n_samples=100, noise=0.2, random_state=42)
title_moons = "Non-linearly Separable Data (Moons)"

# Setup: Split, Scale, and prepare all scaled data
X, y, title = X_moons, y_moons, title_moons
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)

# 1.0 Raw Data Plot (Call the function)
print(f"Plotting: {title} - Raw Data")
plot_svm_results(X, y, title, plot_type='raw')
# sys.exit()

# Prepare a list of kernels and their parameters for iteration
kernels = [
    ('linear', svm.SVC(kernel='linear',  random_state=42)),
    ('poly (d=3)', svm.SVC(kernel='poly', degree=3, random_state=42)),
    ('rbf', svm.SVC(kernel='rbf', random_state=42)),
    ('sigmoid', svm.SVC(kernel='sigmoid', random_state=42))
]

# 1.1 - 1.4 Training and Plotting SVM Results (Loop and call the function)
for kernel_name, clf in kernels:
    print(f"Training and Plotting: {title} - Kernel: {kernel_name}")
    # Train the classifier
    clf.fit(X_train_scaled, y_train)

    # Plot the result using the function
    plot_svm_results(
        X_all_scaled, y, title,
        plot_type='result',
        clf=clf,
        X_test=X_test_scaled,
        y_test=y_test,
        kernel_type=kernel_name
    )
