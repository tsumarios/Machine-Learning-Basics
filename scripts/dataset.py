#!/usr/bin/env python3

"""
    Title:  ML Workshop: Create a Dataset
    Date:   May 8, 2021
    Author: Mario Raciti
"""

import matplotlib.pyplot as plt
import numpy as np

# Vars
obs_n = 200
a, b = 3.5, 8
fluctuations = 0.5

# Features
X = 1 + 2 * np.random.random(obs_n)
# Targets
Y = b + a * X + fluctuations * np.random.randn(obs_n)

# Plot
plt.plot(X, Y, '+')
plt.show()
