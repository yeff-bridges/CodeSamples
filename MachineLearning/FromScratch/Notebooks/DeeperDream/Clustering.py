import numpy as np
from copy import copy
from scipy.stats import multivariate_normal


class K_Means:
    def __init__(self, k):
        self.k = k

    def Fit(self, x):
        self.means = x[np.random.choice(x.shape[0], size=self.k, replace=False), :]
        old_means = None
        while (old_means != self.means).any():
            old_means = self.means
            c = self.Predict(x)
            self.means = np.vstack([np.mean(x[c == k], axis=0) for k in range(self.k)])

    def Predict(self, x):
        return np.argmin(np.sum((x - self.means.reshape([self.k, 1, x.shape[1]])) ** 2, axis=2), axis=0)


class Soft_K_Means:
    def __init__(self, k, beta):
        self.k = k
        self.beta = beta

    def Fit(self, x):
        self.means = x[np.random.choice(x.shape[0], size=self.k, replace=False), :]
        old_means = None
        while (old_means != self.means).any():
            old_means = self.means
            c = self.Predict(x)
            self.means = np.vstack([np.mean(x[c == k], axis=0) for k in range(self.k)])

    def Predict(self, x):
        dist = np.exp(-self.beta * np.sum((x - self.means.reshape([self.k, 1, x.shape[1]]) ** 2), axis=2))
        dist /= np.sum(dist, axis=1)
        return np.argmin(dist, axis=0)


class Gaussian_Mixture:
    def __init__(self, k):
        self.k = k

    def Fit(self, x):
        self.means = x[np.random.choice(x.shape[0], size=self.k, replace=False), :]
        self.covs = [np.eye(x.shape[1])] * self.k
        self.covs = np.stack(self.covs)
        self.norms = [0] * self.k
        old_means = None
        old_covs = None
        i = 0
        while (old_means != self.means).any() or (old_covs != self.covs).any():
            print(f'{i}\r', end='')
            i += 1
            old_means = copy(self.means)
            old_covs = copy(self.covs)
            p_values = []
            for k in range(self.k):
                self.norms[k] = multivariate_normal(mean=self.means[k], cov=self.covs[k])
                p_values.append(self.norms[k].pdf(x))
            p_values = np.vstack(p_values)
            y_hat = np.argmax(p_values, axis=0)

            for k in range(self.k):
                self.means[k, :] = np.mean(x[y_hat == k], axis=0)
                self.covs[k, :, :] = np.cov(x[y_hat == k].T)

    def Predict(self, x):
        p_values = []
        for k in range(self.k):
            p_values.append(self.norms[k].pdf(x))
        p_values = np.vstack(p_values)
        y_hat = np.argmax(p_values, axis=0)
        return y_hat

    def PValues(self, x):
        p_values = []
        for k in range(self.k):
            p_values.append(self.norms[k].pdf(x))
        p_values = np.vstack(p_values)
        return p_values

    def Loss(self, x):
        p_values = self.PValues(x)
        return np.sum(-np.log(np.max(p_values, axis=0)))


def CHI(x, y_hat, means):
    n_k = np.unique(y_hat, return_counts=True)[1]
    top = np.sum(n_k * np.sum((means - np.mean(x, axis=0)) ** 2, axis=1))
    bot = 0
    k = means.shape[0]
    for i in range(k):
        bot += np.sum((x[y_hat == i] - means[i]) ** 2)
    return top * (x.shape[0] - k) / (bot * (k - 1))


def DBI(x, y_hat, means):
    unique, counts = np.unique(y_hat, return_counts=True)
    out = 0
    for k in range(means.shape[0]):
        sig_k = (1 / counts[k]) * np.sum((x[y_hat == k] - means[k]) ** 2)
        max_inner = -np.inf
        for j in range(means.shape[0]):
            if j != k:
                sig_j = (1 / counts[j]) * np.sum((x[y_hat == j] - means[j]) ** 2)
                inner = (sig_j + sig_k) / np.sum((means[j] - means[k]) ** 2)
                if inner > max_inner:
                    max_inner = inner

        out += max_inner
    return out / means.shape[0]


def Silhouette(x, y_hat, means):
    b = np.array(
        [np.mean(np.sqrt(np.sum((x[i == y_hat] - means[i]) ** 2, axis=1)), axis=0) for i in range(means.shape[0])])
    a = np.mean(
        [np.sqrt(np.sum((np.dstack([x] * means.shape[0]) - means.T) ** 2, axis=1))[i, range(means.shape[0]) != yi]
         for i, yi in enumerate(y_hat)],
        axis=1)

    return np.array([(a[i] - b[yi]) / np.maximum(a[i], b[yi]) for i, yi in enumerate(y_hat)])