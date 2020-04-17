"""

"""
import numpy as np


def Sigmoid(z):
    """
    Computes the sigmoid for z

    :param z: nd array
    :return: 1 / (1 + e^{-z})
    """
    return 1 / (1 + np.exp(-z))


def SoftMax(z):
    """
    Computes the softemax for z

    :param z: nd array
    :return: e^z / Sum(e^{z_i})
    """
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

