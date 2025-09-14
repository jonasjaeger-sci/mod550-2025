import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from DataAkquisitionT2 import DataAkquisition

rnd_data = DataAkquisition(nRows=10)
rnd_data.uniform_dist()
#rnd_data.gaussian_dist()
ax = sb.heatmap(rnd_data.data,annot=True,linewidths=0.5,cmap="crest")
ax.set_xlabel("Features")
ax.set_ylabel("Experiments")
plt.show()

