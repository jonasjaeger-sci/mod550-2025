import numpy as np

probabilities = np.array([0.01, 0.24, 0.05, 0.13, 0.16, 0.31, 0.1])
outcomes = np.array(list(range(7)))

ind_entropy = probabilities * np.log(1/probabilities)
print(ind_entropy)
entropy = np.sum(ind_entropy)
print(f"The Entropy of the discrete distribution is: {entropy}")