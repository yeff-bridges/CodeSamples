import numpy as np

def gini(y):
    labels, freq = np.unique(y, return_counts=True)
    t = freq / np.sum(freq)
    return np.sum(t*(1-t))