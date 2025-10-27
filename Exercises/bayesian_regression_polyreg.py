import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sys

# Reproducibility
rng = np.random.default_rng(42)

# Parameters for the synthetic data
N = 60
true_intercept = 5.0
true_slope = 1.
true_quad_slope = 0.1
sigma_true = 20.0  # controls scatter

age = rng.integers(18, 70, size=N)
#print(f"age:{age}")
time_spent = (true_intercept + true_slope*age + true_quad_slope*age**2 +
              rng.normal(0, sigma_true, size=N))

data = (pd.DataFrame({'Age': age, 'Time': time_spent})
        .sort_values('Age').reset_index(drop=True))
#print(data.head())

# visualize synthetic data
"""
plt.figure(figsize=(5,5))
plt.scatter(data['Age'], data['Time'], s=18)
plt.xlabel('Age (years)')
plt.ylabel('Time on App (minutes)')
plt.title('Scatter: Time vs Age')
plt.grid(True)
plt.show()
"""

# perform traditional frequentist linear regression
X = data["Age"].values.reshape(-1,1)
y = data["Time"].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
polyreg = LinearRegression().fit(X_poly,y)
polyreg_predict = polyreg.predict(X_poly)
polyreg_intercept = float(polyreg.intercept_)
polyreg_slope = float(polyreg.coef_[1])
polyreg_quad_slope = float(polyreg.coef_[2])

print(f'polyreg intercept  : {polyreg_intercept:.4f}')
print(f'polyreg slope      : {polyreg_slope:.4f}')
print(f'polyreg quad slope : {polyreg_quad_slope:.4f}')

age_grid = np.linspace(data['Age'].min(), data['Age'].max(), 200)
polyreg_line = polyreg_intercept + polyreg_slope * age_grid + polyreg_quad_slope*age_grid**2

# visualize frequentist linear regression with
# ordinary least squares
"""
plt.figure(figsize=(5,5))
plt.scatter(data['Age'], data['Time'], s=18)
plt.plot(X, polyreg_predict, linewidth=2, color = 'red')
plt.plot(age_grid, polyreg_line, linewidth=2, linestyle="--" ,color = 'green')
plt.xlabel('Age (years)')
plt.ylabel('Time on App (minutes)')
plt.title('frequentist LR Fit')
plt.show()
"""

# perform bayesian linear regression
# 1) Make sure arrays are 1-D float
X = data["Age"].to_numpy(dtype=float)
Y = data["Time"].to_numpy(dtype=float)

# 2) Bayesian linear regression with weak priors
with pm.Model() as model:
    b0    = pm.Normal("b0", mu=0, sigma=100)
    b1    = pm.Normal("b1", mu=0, sigma=50)
    b2    = pm.Normal("b2", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)

    mu = b0 + b1 * X + b2 * X**2
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y)

    #  Correct Monte Carlo (SMC) sampler — no tune/target_accept
    trace = pm.sample_smc(draws=4000, progressbar=True)


# 3) Extract posterior samples
post_b0    = trace.posterior["b0"].values.ravel()
post_b1    = trace.posterior["b1"].values.ravel()
post_b2    = trace.posterior["b2"].values.ravel()
post_sigma = trace.posterior["sigma"].values.ravel()
backend = "PyMC (SMC Monte Carlo)"

# visualize bayesian regression
plt.figure(figsize=(5,5))
plt.scatter(data['Age'], data['Time'], s=18)

idx = np.random.default_rng(7).choice(len(post_b0), size=300,
                                      replace=False)
for i in idx:
    plt.plot(age_grid, post_b0[i] + post_b1[i]*age_grid + post_b2[i]*age_grid**2,
             alpha=0.06)

"""
plt.plot(age_grid, polyreg_line, linewidth=2, color = 'red')  # OLS reference
plt.xlabel('Age (years)')
plt.ylabel('Time on App (minutes)')
plt.title('Cloud of Bayesian Lines vs OLS')
plt.tight_layout()
plt.show()
"""

# Posterior distributions and credible intervals
def ci( a,alpha=0.05):
    """
    function to calculate low and high values of a dataset within 1-alpha interval
    Parameters
    ----------
    a: float array
        posterior values
    alpha: float
        value defining the interval

    Returns
    -------
    lo: float
        lowest posterior value within interval
    hi: float
        highest posterior value within interval
    """
    lo = np.quantile(a,alpha/2)
    hi = np.quantile(a,1-alpha/2)

    return lo, hi

b1_mean = float(np.mean(post_b1))
b1_lo, b1_hi = ci(post_b1)
b2_mean = float(np.mean(post_b2))
b2_lo, b2_hi = ci(post_b2)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_xlabel("Slope")
ax.set_ylabel("Frequency")
ax.set_title(f'Posterior Slope \nMean={b1_mean:.3f}, 95% CI=({b1_lo:.3f}, {b1_hi:.3f})')
ax.hist(post_b1,bins=50)
ax.axvline(polyreg_slope,linewidth=2, color="red")

ax = fig.add_subplot(1,2,2)
ax.set_xlabel("quadratic Slope")
ax.set_ylabel("Frequency")
ax.set_title(f'Posterior quadratic Slope \nMean={b2_mean:.3f}, 95% CI=({b2_lo:.3f}, {b2_hi:.3f})')
ax.hist(post_b2,bins=50)
ax.axvline(polyreg_quad_slope,linewidth=2, color="red")

plt.show()