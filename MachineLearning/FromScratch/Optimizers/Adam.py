import numpy as np

class Adam:
    def __init__(self, parameters, learn_rate=1e-2, k_1=0.5, k_2=0.5):
        self.parameters = parameters
        self.learn_rate = learn_rate
        self.k_1 = k_1
        self.k_2 = k_2
        self.m = [np.zeros(param['array'].shape) for param in parameters]
        self.v = [np.zeros(param['array'].shape) for param in parameters]
        self.t = 0

    def update_weights(self):
        self.t += 1
        for idx, param in enumerate(self.parameters):
            self.m[idx] = self.k_1 * self.m[idx] + (1-self.k_1) * param['grad']
            self.v[idx] = self.k_2 * self.v[idx] + (1-self.k_2) * param['grad']**2
            m_hat = self.m[idx] / (1 - self.k_1**self.t)
            v_hat = self.v[idx] / (1 - self.k_2**self.t)
            param['array'] -= self.learn_rate * m_hat / np.sqrt(v_hat + 1e-99)