import numpy as np

def Binary_Cross_Entropy(y, p_hat):
    '''
    :param y:
    :param p_hat:
    :return:
    '''
    return -1 * np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

def MSE(y, y_hat):
    '''
    :param y:
    :param y_hat:
    :return: Mean squared error
    '''
    return np.trace((y - y_hat).T @ (y - y_hat)) / y.shape[0]

def Cross_Entropy(y, p_hat):
    '''
    :param y:
    :param p_hat:
    :return:
    One-Hot Encoded
    '''
    return -np.sum(y * np.log(p_hat))