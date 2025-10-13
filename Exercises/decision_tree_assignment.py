"""

"""
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from DataGenerator import DataGenerator


# generate gaussian distribution
# feature: body height
n_datapoints = 1000
mean = 175 # average size
std = 15
sampler = DataGenerator(n_rows=n_datapoints)
sampler.gaussian_dist(mean=mean, std=std)
s1_gaussian = np.round(sampler.data[:,0],1)

# generate uniform distribution
# feature: age
min_age = 15; max_age = 95
sampler.uniform_dist(lb=min_age, ub=max_age)
s2_uniform = np.round(sampler.data[:,0],1)

# generate sinusoidal distribution + noise
# feature: systolic blood pressure
base = 120
dev = 30
n_periods = 2
sampler.sine(A=dev, C=base, k=n_periods)
sampler.add_uniform_noise(lb = 10, ub = -10)
s3_sine = np.round(sampler.data[:,1],1)
#plt.plot(s3_sine)
#plt.show()

# consolidate features into single array
X = np.column_stack([s1_gaussian, s2_uniform, s3_sine],)
print(X.shape)

# generate label Y
smoker_class = ["non-smoker", "socially", "daily", "chain-smoker"]
Y = np.random.choice(smoker_class, size=n_datapoints)

print(Y.shape)

plt.figure(figsize=(10,5), dpi= 200)
clf = DecisionTreeClassifier(max_leaf_nodes=10, criterion="entropy")

clf.fit(X, Y)
plot_tree(clf, proportion=True, filled=True)

plt.show()



"""
# generate gaussian distribution
# feature: size
mean = 175 # average size
std = 15
sampler = DataGenerator()
sampler.gaussian_dist(mean=mean, std=std)
s1_gaussian = np.round(sampler.data[:,0],1)

n_bins = 50
delta = (max(s1_gaussian)-min(s1_gaussian))/n_bins
bins = np.arange(min(s1_gaussian), max(s1_gaussian+delta), delta)
#print(bins)

hist, edges = np.histogram(s1_gaussian, bins= bins)
dist1_gauss = hist/sum(hist)
s1_gaussian = (edges[1:] + edges[:-1]) * 0.5
#plt.plot(s1_gaussian,dist1_gauss)
#plt.show()

# generate random distribution
# feature: age
min_age = 15; max_age = 95
sampler.uniform_dist(lb=min_age, ub=max_age)
s2_uniform = np.round(sampler.data[:,0],1)
delta = (max(s2_uniform)-min(s2_uniform))/n_bins
bins = np.arange(min(s2_uniform), max(s2_uniform+delta), delta)

hist, edges = np.histogram(s2_uniform, bins= bins)
dist2_uniform = hist/sum(hist)
s2_uniform = (edges[1:] + edges[:-1]) * 0.5
#plt.plot(s2_uniform,dist2_uniform)
#plt.show()

# generate exponential distribution
sampler.exponential()
sampler.add_gaussian_noise()

# generate label
smoker_class = ["non-smoker", "socially", "daily", "chain-smoker"]
label = np.random.choice(smoker_class,1000)

print(label)

"""


