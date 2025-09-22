"""
Data Processing Task 5 to calculate probability mass function + cumulative distribution function
"""

import numpy as np
import matplotlib.pyplot as plt
from DataAcquisitionT5 import DataAcquisition

# read data from .csv-file
datareader = DataAcquisition()
hourly_price_MWh_24 = datareader.read_csv(path="../Task3/hourly_price_MWh_20240101_20250101.csv",
                                          delimiter=";")
#print(hourly_price_MWh_24.columns)
hourly_price_MWh_24_DE = hourly_price_MWh_24['Deutschland/Luxemburg [€/MWh] Originalauflösungen']
#print(hourly_price_MWh_24_DE)

min_price = hourly_price_MWh_24_DE.min()
max_price = hourly_price_MWh_24_DE.max()

NUM_BINS = 50
binwidth = abs(max_price - min_price) / NUM_BINS
bins = np.arange(int(min_price), int(max_price) + binwidth, binwidth)
bin_centers = (bins[:-1] + bins[1:])*0.5
#print(bin_centers)

# calculate probability mass function + cumulative distribution function
# discrete
pmf_discrete = hourly_price_MWh_24_DE.value_counts(normalize=True).sort_index()
#print(pmf_discrete)
cmf_discrete = np.cumsum(pmf_discrete)
#print(cmf_discrete)

# continuous
numOccurences_continuous , edges = np.histogram(hourly_price_MWh_24_DE , bins=bins)
pmf_continuous = numOccurences_continuous/numOccurences_continuous.sum()
#print(pmf_continuous)
cmf_continuous = np.cumsum(pmf_continuous)
#print(cmf_continuous)


#"""
# plot 2D-histogram and pmf
fig1 = plt.figure()

ax2 = fig1.add_subplot(1,2,1)
ax2.bar(pmf_discrete.index , pmf_discrete.values , color="red")
ax2.set_xlabel("Price [€/MWh]" , fontsize=14)
ax2.set_ylabel("Probability" , fontsize=14 , color="red")
#ax2.set_xticks(np.linspace(min_price,max_price,15))
ax2.set_title("discrete probability mass function" , fontsize=16)
ax2.grid(True)

ax2_2 = ax2.twinx()
ax2_2.plot(cmf_discrete.index , cmf_discrete.values, linewidth=3.0 , color="blue")
ax2_2.set_ylabel("cumulative probability", fontsize=14 , color="blue")

ax3 = fig1.add_subplot(1,2,2)
ax3.bar(bin_centers , pmf_continuous , width = np.diff(bins), color="red" , edgecolor="black")
ax3.set_xlabel("Price [€/MWh]" , fontsize=14)
ax3.set_ylabel("Probability" , fontsize=14 , color="red")
#ax3.set_xticks(np.linspace(min_price,max_price,15))
ax3.set_title("continuous probability mass function" , fontsize=16)
ax3.grid(True)

ax3_2 = ax3.twinx()
ax3_2.plot(bin_centers,cmf_continuous, linewidth=3.0 , color="blue")
ax3_2.set_ylabel("cumulative probability", fontsize=14 , color="blue")

plt.show()
#"""
