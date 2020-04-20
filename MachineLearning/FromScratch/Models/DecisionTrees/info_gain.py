import numpy as np

def info_gain(y, mask, imp_func):
    gini = imp_func(y[y.columns[0]])
    imps = []
    unique, counts = np.unique(mask, return_counts=True)
    for i in unique:
        imps.append(imp_func(y[y.columns[0]][mask == i]))
    gain = gini - (np.dot(counts, imps)) / y.shape[0]
    return gain