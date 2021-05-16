#!/usr/bin/env python3

"""
    Title:  ML Workshop: Simple Linear Regression (SLR)
    Date:   May 15, 2021
    Author: Mario Raciti
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

np.random.seed(0)

# Create a dataset
# Vars
obs_n = 200
a, b = 3.5, 8
fluctuations = 1
# Feature
X = 1 + 2 * np.random.random(size=(obs_n, 1))
# Target
Y = b + a * X.squeeze() + fluctuations * np.random.randn(obs_n)
# Plot dataset to chart
plt.plot(X, Y, "+")
plt.show()

# Create an SLR model
slr_model = LinearRegression()
# Training
slr_model.fit(X, Y)
print(slr_model.coef_)
print(slr_model.intercept_)

# Predictions
# Single-point prediction
X_pred = np.array([2, ])[:, np.newaxis]
# Multiple-point prediction
X_pred = np.array([2, 3])[:, np.newaxis]
# Multiple-point prediction (dynamic)
first, last, new_obs = X.min(), X.max(), 100
X_pred = np.linspace(first, last, new_obs)[:, np.newaxis]

# Predict
Y_pred = slr_model.predict(X=X_pred)

# Plot predictions to chart
plt.plot(X, Y, '+')
plt.plot(X_pred, Y_pred, 'r-')
plt.show()

# Model Evaluation
# K-fold Cross Validation
K = 5
print(f'Number of folds = {K}\n')
error = 'neg_mean_squared_error'    # NOTE: MSE = RSS/N
results = cross_validate(slr_model, X, Y, cv=K, scoring=error)

# Error for each fold and average error over the folds:
errors = -results['test_score']
print(f'Error for each fold: {errors}')
print(f'Error averaged on all folds: {errors.mean()}')
