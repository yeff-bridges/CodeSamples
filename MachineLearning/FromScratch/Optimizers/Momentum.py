import numpy as np

class Momentum:
    def __init__(self, parameters, learn_rate=1e-4, mu=0.2):
        self.learn_rate = learn_rate
        self.mu = mu
        self.parameters = parameters
        self.v = [np.zeros(param['array'].shape) for param in self.parameters]

    def update_weights(self):
        for i, param in enumerate(self.parameters):
            self.v[i] = self.mu * self.v[i] - self.learn_rate * param['grad']
            param['array'] += self.v[i]