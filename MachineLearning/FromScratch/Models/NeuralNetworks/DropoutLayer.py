import numpy as np

class DropoutLayer(BaseLayer):
    """
    A BaseLayer which randomly zeros a portion of the outputs of the BaseLayer coming before it.
    The shape of the input is preserved, and the random zeroing is only performed during training
    """
    def __init__(self, drop_rate=.5):
        super().__init__()
        self.drop_rate = drop_rate

    def backward(self, z_in, y):
        mask = np.random.rand(np.shape(z_in)[0], np.shape(z_in)[1]) > self.drop_rate
        z_out = mask * z_in

        grad_after = self.after.backward(z_out, y)
        grad_out = grad_after * mask

        return grad_out

    def forward(self, z_in):
        mask = np.random.rand(np.shape(z_in)[0], np.shape(z_in)[1]) > self.drop_rate
        z_out = mask * z_in
        return z_out

    def predict(self, z_in):
        return self.after.predict(z_in)

    def get_parameters(self):
        parameters = self.after.get_parameters()
        return parameters