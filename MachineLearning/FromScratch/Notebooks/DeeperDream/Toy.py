"""
Used to make toy data sets
"""

import numpy as np


def Donut(n, r, margin):
    x = np.random.randn(n, 2)
    x_donut = x[np.sqrt(np.sum(x ** 2, axis=1)) > 1] * (r + margin / 2)
    x_hole = x[np.sqrt(np.sum(x ** 2, axis=1)) <= 1] * (r - margin / 2)

    y_hole = np.zeros([x_hole.shape[0], 1])
    y_donut = np.ones([x_donut.shape[0], 1])

    x = np.vstack([x_hole, x_donut])
    y = np.vstack([y_hole, y_donut])
    return x, y


def Clusters(n, cats, dims, spread):
    x = []
    y = []
    for i in range(cats):
        x.append(np.random.randn(n, dims) + np.random.randn(dims)*spread)
        y.append(np.array([np.arange(cats)] * n) == i)
    x = np.vstack(x)
    y = np.vstack(y)
    return x, y


def CovClusters(n, cats, dims, spread):
    x = []
    y = []
    for i in range(cats):
        cov = np.random.rand(dims, dims)
        cov = cov @ cov.T * 2 - 1
        x.append(np.random.randn(n, dims) @ cov + np.random.randn(dims)*spread)
        y.append(np.array([np.arange(cats)] * n) == i)
    x = np.vstack(x)
    y = np.vstack(y)
    return x, y
