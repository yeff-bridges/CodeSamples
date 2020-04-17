import numpy as np


class Linear:
    def __call__(self, z):
        return z

    def D(self, z):
        return 1


class Sigmoid:
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def D(self, z):
        return self(z) * (1 - self(z))


class SoftMax:
    def __call__(self, z):
        return np.exp(z) / (np.sum(np.exp(z), axis=1, keepdims=True) + 1e-99)


class ReLU:
    def __call__(self, z):
        return z * (z > 0)

    def D(self, z):
        return z > 0


class Tanh:
    def __call__(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def D(self, z):
        return 1 - (self(z) ** 2)


class LeakyReLU:
    def __init__(self, leak_coeff):
        self.leak = leak_coeff

    def __call__(self, z):
        return z * (z > 0) + self.leak * z * (z <= 0)

    def D(self, z):
        return 1 * (z > 0) + self.leak * (z <= 0)