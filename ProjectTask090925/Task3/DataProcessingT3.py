import numpy as np
import matplotlib.pyplot as plt
from DataAcquisitionT3 import DataAkquisition

# read data from .csv-file
datareader = DataAkquisition()
hourly_price_MWh_24 = datareader.read_csv(path="hourly_price_MWh_20240101_20250101.csv",delimiter=";")
#print(hourly_price_MWh_24.columns)
hourly_price_MWh_24_DE = hourly_price_MWh_24['Deutschland/Luxemburg [€/MWh] Originalauflösungen']
#print(hourly_price_MWh_24_DE)
min_price = hourly_price_MWh_24_DE.min()
max_price = hourly_price_MWh_24_DE.max()

numBins = 100
binwidth = abs(max_price - min_price)/numBins
bins = np.arange(int(min_price), int(max_price) + binwidth, binwidth)

#""""
# plot 2D-histogram
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.hist(hourly_price_MWh_24_DE , bins=bins)
ax1.set_xlabel("Price [€/MWh]",fontsize=14)
ax1.set_ylabel("Number of hourly occurrances per year",fontsize=14)
ax1.set_xticks(np.linspace(min_price,max_price,15))
#ax1.grid(True)

plt.show()
#"""