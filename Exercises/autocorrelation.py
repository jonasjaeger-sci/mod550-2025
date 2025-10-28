import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

# generate synthetic timeseries
np.random.seed(14)
n = 100
time = np.arange(n)
trend = 0.4 * time
seasonality = 2* np.sin(2*np.pi * time/12)
noise = np.random.uniform(-3.14,3.14,n)
data = trend + seasonality + noise
ts = pd.Series(data)

# plot generated data
plt.figure()
plt.plot(ts)
plt.title("Generated Time Sereies")
plt.xlabel("time")
plt.ylabel("Value")
plt.grid(True)
#plt.show()

# calculate auto correlation function and plot
plt.figure()
acf_vals = acf(ts,nlags=100)
plt.stem(range(len(acf_vals)), acf_vals)
plt.title("Autocorrelation Function ACF")
plt.xlabel("lag")
plt.ylabel("rho")
plt.grid(True)
#plt.show()

# fit ARIMA model
# the order should be obtained from running PACF, ACF and differencing
model1 = ARIMA(ts, order=(1, 1, 1))
result1 = model1.fit()

model2 = ARIMA(ts, order=(0, 1, 0))
result2 = model2.fit()

# plot
plt.figure()
plt.plot(ts, label="original series", color="gray")
plt.plot(result1.fittedvalues, label="ARIMA-fitted model 1", color="red")
plt.plot(result2.fittedvalues, label="ARIMA-fitted model 2", color="blue")
plt.title("ARIMA-fit vs. original series")
plt.xlabel("time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# compare different models
print(result1.summary())
print(result2.summary())

pred = result1.predict(start=len(ts), end = len(ts)+20)
print(f"predictions:\n {pred}")
plt.show()