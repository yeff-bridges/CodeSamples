import numpy as np

def Binary_Cross_Entropy(y, p_hat):
    '''
    :param y: target vector (zeros or ones)
    :param p_hat: prediction vector (as probabilities)
    :return: negative log-likelihood loss for all targets
    '''
    return -1 * np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

def MSE(y, y_hat):
    '''
    :param y: target vector
    :param y_hat: prediction vector
    :return: Mean squared error for all targets
    '''
    return np.trace((y - y_hat).T @ (y - y_hat)) / y.shape[0]

def Cross_Entropy(y, p_hat):
    '''
    :param y: target vector (multi-class, one-hot encoded)
    :param p_hat: prediction vectors (probability distributions)
    :return: negative log-likelihood loss for all targets
    '''
    return -np.sum(y * np.log(p_hat))

def Sparse_Cross_Entropy(y, p_hat):
    '''
    :param y: target vector (multi-class, integer category)
    :param p_hat: prediction vectors (probability distributions)
    :return: negative log-likelihood loss for all targets
    '''
    return -np.sum(np.log(p_hat[y]))
