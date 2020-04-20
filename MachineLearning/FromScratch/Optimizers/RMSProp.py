import numpy as np

class RMSProp:
    def __init__(self, parameters, learn_rate=1e-2, k=0.5):
        self.parameters = parameters
        self.learn_rate = learn_rate
        self.k = k
        self.g = [np.zeros(param['array'].shape) for param in parameters]

    def update_weights(self):
        for idx, param in enumerate(self.parameters):
            self.g[idx] = self.k * self.g[idx] + (1 - self.k) * param['grad']**2
            param['array'] -= param['grad'] * self.learn_rate/np.sqrt(self.g[idx] + 1e-99)