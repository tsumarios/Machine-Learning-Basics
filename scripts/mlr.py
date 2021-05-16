#!/usr/bin/env python3

"""
    Title:  ML Workshop: Multiple Linear Regression (MLR)
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
b0, b1, b2, b3 = 5, 8, -5, 0.1
fluctuations = 1
# Features
X1 = 1 + 2 * np.random.random(size=(obs_n, 1))  # in (1, 3)
X2 = 2 * np.random.random(size=(obs_n, 1))      # in (0, 2)
X3 = 4 + 3 * np.random.random(size=(obs_n, 1))  # in (4, 7)
X = np.concatenate((X1, X2, X3), axis=1)    # Gather features
# # Plot features to chart
# plt.plot(X1.squeeze(), "r+")
# plt.plot(X2.squeeze(), "b*")
# plt.plot(X3.squeeze(), "go")
# plt.suptitle('X1, X2, X3', fontsize=16)
# plt.show()
# Targets
Y = b0 + b1 * X1.squeeze() + b2 * X2.squeeze() + b3 * X3.squeeze() + \
    fluctuations * np.random.randn(obs_n)
# # Plot dataset to chart
# plt.plot(X1.squeeze(), Y, 'b+')
# plt.suptitle('Y vs X1', fontsize=16)
# plt.show()
# plt.plot(X2.squeeze(), Y, 'b+')
# plt.suptitle('Y vs X2', fontsize=16)
# plt.show()
# plt.plot(X3.squeeze(), Y, 'b+')
# plt.suptitle('Y vs X3', fontsize=16)
# plt.show()

# Create an MLR model
mlr_model = LinearRegression()
# Training
mlr_model.fit(X, Y)
print(mlr_model.coef_)
print(mlr_model.intercept_)
# Results
print(f'Source b0 = {b0}\t Model b0 = {mlr_model.intercept_:.2f}\tRelative difference {100*(b0-mlr_model.intercept_)/mlr_model.intercept_:.2f}%')
print(f'Source b1 = {b1}\t Model b1 = {mlr_model.coef_[0]:.2f}\tRelative difference {100*(b1-mlr_model.coef_[0])/mlr_model.coef_[0]:.2f}%')
print(f'Source b2 = {b2}\t Model b2 = {mlr_model.coef_[1]:.2f}\tRelative difference {100*(b2-mlr_model.coef_[1])/mlr_model.coef_[1]:.2f}%')
print(f'Source b3 = {b3}\t Model b3 = {mlr_model.coef_[2]:.2f}\tRelative difference {100*(b3-mlr_model.coef_[2])/mlr_model.coef_[2]:.2f}%')

# Predictions
new_obs = 100
i = 0
feature_n = np.shape(X)[1]
X_pred = np.empty([new_obs, feature_n])
for feature in [X1, X2, X3]:
    first, last = feature.min(), feature.max()
    X_pred[:, i] = np.linspace(first, last, new_obs)
    i += 1

# Predict
Y_pred = mlr_model.predict(X=X_pred)

# Plot predictions to chart
plt.plot(X, Y, '+')
plt.plot(X_pred, Y_pred, 'r-')
plt.show()

# Model Evaluation
# K-fold Cross Validation
K = 5
print(f'Number of folds = {K}\n')
error = 'neg_mean_squared_error'    # NOTE: MSE = RSS/N
results = cross_validate(mlr_model, X, Y, cv=K, scoring=error)

# Error for each fold and average error over the folds:
errors = -results['test_score']
print(f'Error for each fold: {errors}')
print(f'Error averaged on all folds: {errors.mean()}')
