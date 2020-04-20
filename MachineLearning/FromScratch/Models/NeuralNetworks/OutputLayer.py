import numpy as np

class OutputLayer(Layer):
    """
    Similar to Layer, but this one doesn't assume it has as BaseLayer after it
    """
    def backward(self, z_in, y):
        y_hat = self.forward(z_in)
        grad_h = y_hat - y

        grad_w = z_in.T @ grad_h
        grad_b = np.sum(grad_h, axis=0)
        grad_z_in = grad_h @ self.w["array"].T

        self.w["grad"] = grad_w
        self.b["grad"] = grad_b

        return grad_z_in

    def predict(self, z_in):
        return self.forward(z_in)

    def get_parameters(self):
        my_parameters = [self.w, self.b]
        return my_parameters