import numpy as np

class BatchNormLayer(BaseLayer):
    """
    A BaseLayer which normalizes the outputs of the previous BaseLayer based on a moving average of all outputs seen during training.
    This normalization preserves the input shape, and the moving average is only updated during training
    """
    def __init__(self, norm_p):
        super().__init__()
        self.mu = 0
        self.sig = 1
        self.norm_p = norm_p

    def forward(self, z_in):
        mu = np.mean(z_in, keepdims=True, axis=0)
        self.mu = self.norm_p * self.mu + (1 - self.norm_p) * mu
        sig = np.std(z_in, keepdims=True, axis=0)
        self.sig = self.norm_p * self.sig + (1 - self.norm_p) * sig

        z_out = (z_in - self.mu) / (self.sig + 1e-99)
        return z_out

    def backward(self, z_in, y):
        z_out = self.forward(z_in)

        grad_after = self.after.backward(z_out, y)
        grad_out = grad_after / self.sig
        return grad_out

    def predict(self, z_in):
        mu = np.mean(z_in, keepdims=True, axis=0)
        mu = self.norm_p * self.mu + (1 - self.norm_p) * mu
        sig = np.std(z_in, keepdims=True, axis=0)
        sig = self.norm_p * self.sig + (1 - self.norm_p) * sig

        z_out = (z_in - mu) / (sig + 1e-99)
        return self.after.predict(z_out)

    def get_parameters(self):
        parameters = self.after.get_parameters()
        return parameters