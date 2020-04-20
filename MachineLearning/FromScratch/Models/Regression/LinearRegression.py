from matplotlib import pyplot as plt
import numpy as np

from FromScratchModule.Activation.Function import *
from FromScratchModule.Loss import mean_squared_error, binary_cross_entropy, cross_entropy

class LinearRegression:
    def __init__(self, num_iter=1000, tol=1e-4, learning_rate=1e-4, alpha=0, beta=0.5):
        self.num_iter = num_iter
        self.tolerance = tol
        self.learning_rate = learning_rate
        self.w = None
        self.alpha = alpha
        self.beta = beta

    def fit(self, x, y, pad=False, plot_loss=False):
        """
        This method get x and y nd arrays and apply the gradient descent method.
        :param plot_loss:
        :param x: nd array
        :param y: nd array
        :param pad: boolean argument to add y-intercept
        :return: self.beta
        """
        loss = [np.inf]

        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        self.w = np.random.randn(x.shape[1], y.shape[1])

        for i in range(self.num_iter):
            y_hat = x @ self.w
            loss.append(np.trace((y - y_hat).T @ (y - y_hat) / y.shape[0]))
            grad = -x.T @ (y - y_hat) + \
                   self.alpha * (
                           self.beta * np.sign(self.w) +  # L1
                           (1 - self.beta) * self.w  # L2
                   )
            self.w -= self.learning_rate * grad
            if abs(loss[-1] - loss[-2]) < self.tolerance:
                break
        if plot_loss:
            plt.plot(loss)
        return self.w

    def batch_fit(self, x, y, pad=False):
        """
        This method get x and y nd arrays and apply the batch gradient descent method.
        Doesn't initialize the weights every time so we can feed it different data points.
        Only runs one iteration.
        :param x: nd array
        :param y: nd array
        :param pad: boolean argument to add y-intercept
        :return: self.beta
        """

        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        if self.w is None:  # Checks to see if we already have weights
            self.w = np.random.randn(x.shape[1], y.shape[1])

        y_hat = self.predict(x)
        grad = -x.T @ (y - y_hat) + self.alpha * (
                self.beta * np.sign(self.w) +  # L1
                (1 - self.beta) * self.w  # L2
        )

        self.w -= self.learning_rate * grad
        loss = mean_square_error(y, y_hat)
        return loss

    def predict(self, x, pad=False):
        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        y_hat = x @ self.w
        return y_hat