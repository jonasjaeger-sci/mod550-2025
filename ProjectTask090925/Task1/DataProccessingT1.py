from DataAkquisitionT1 import DataAkquisition
import matplotlib.pyplot as plt

# generate data instance
rnd_data = DataAkquisition()
#rnd_data.uniform_dist(lb=0 , ub=10)
rnd_data.gaussian_dist()

# plot in 2D-histogram
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.hist(rnd_data.data[:,1])
ax.set_xlabel("bins")
ax.set_ylabel("number of elements in bin")
#ax.grid(True)
plt.show()