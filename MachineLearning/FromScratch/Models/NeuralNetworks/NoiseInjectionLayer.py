import numpy as np

class NoiseInjectionLayer(BaseLayer):
    """
    A BaseLayer which adds random noise to the outputs of the BaseLayer fed into it
    The shape of the inputs is preserved, and the noise is only injected during training
    """
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, z_in):
        mu = np.mean(z_in, keepdims=True, axis=0)
        std = np.std(z_in, keepdims=True, axis=0)
        z_out = z_in + (np.random.randn(z_in.shape[0], z_in.shape[1]) * std + mu) * self.sigma
        return z_out

    def backward(self, z_in, y):
        z_out = self.forward(z_in)

        grad_out = self.after.backward(z_out, y)
        return grad_out

    def predict(self, z_in):
        return self.after.predict(z_in)

    def get_parameters(self):
        parameters = self.after.get_parameters()
        return parameters