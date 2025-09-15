import numpy as np
import matplotlib.pyplot as plt
from DataAcquisitionT4 import DataAkquisition

# read data from .csv-file
datareader = DataAkquisition()
hourly_price_MWh_24 = datareader.read_csv(path="../Task3/hourly_price_MWh_20240101_20250101.csv",delimiter=";")
#print(hourly_price_MWh_24.columns)
hourly_price_MWh_24_DE = hourly_price_MWh_24['Deutschland/Luxemburg [€/MWh] Originalauflösungen']
#print(hourly_price_MWh_24_DE)

min_price = hourly_price_MWh_24_DE.min()
max_price = hourly_price_MWh_24_DE.max()

numBins = 100
binwidth = abs(max_price - min_price)/numBins
bins = np.arange(int(min_price), int(max_price) + binwidth, binwidth)
bin_centers = (bins[:-1] + bins[1:])*0.5
#print(bin_centers)

# calculate probability mass function
# discrete
#numOccurences_discrete = hourly_price_MWh_24_DE.value_counts()
#print(numOccurences_discrete)
#pmf_discrete = numOccurences_discrete/numOccurences_discrete.sum()
pmf_discrete = hourly_price_MWh_24_DE.value_counts(normalize=True).sort_index()
#print(pmf_discrete)


# continuous
numOccurences_continuous , edges = np.histogram(hourly_price_MWh_24_DE , bins=bins)
pmf_continuous = numOccurences_continuous/numOccurences_continuous.sum()
#print(pmf_continuous)


#"""
# plot 2D-histogram and pmf
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
ax1.hist(hourly_price_MWh_24_DE , bins=bins)
ax1.set_xlabel("Price [€/MWh]",fontsize=14)
ax1.set_ylabel("Number of hourly occurrances per year",fontsize=14)
ax1.set_xticks(np.linspace(min_price,max_price,15))
ax1.set_title("2D-histogram")
ax1.grid(True)

ax2 = fig1.add_subplot(1,3,2)
ax2.bar(pmf_discrete.index , pmf_discrete.values)
ax2.set_xlabel("Price [€/MWh]" , fontsize=14)
ax2.set_ylabel("Probability" , fontsize=14)
#ax2.set_xticks(np.linspace(min_price,max_price,15))
ax2.set_title("discrete probability mass function")
ax2.grid(True)

ax3 = fig1.add_subplot(1,3,3)
ax3.bar(bin_centers , pmf_continuous , edgecolor="black")
ax3.set_xlabel("Price [€/MWh]" , fontsize=14)
ax3.set_ylabel("Probability" , fontsize=14)
#ax3.set_xticks(np.linspace(min_price,max_price,15))
ax3.set_title("continuous probability mass function")
ax3.grid(True)

plt.show()
#"""
