import numpy as np


class GD:
    def __init__(self, parameters, learn_rate = 1e-3):
        self.learn_rate = learn_rate
        self.parameters = parameters

    def UpdateWeights(self):
        for parameter in self.parameters:
            parameter['array'] -= parameter['grad'] * self.learn_rate


class Exp_Decay:
    def __init__(self, parameters, learn_rate=1e-3, k=0.01):
        self.learn_rate = learn_rate
        self.k = k
        self.t = 0
        self.parameters = parameters

    def UpdateWeights(self):
        learn_rate = self.learn_rate * np.exp(-self.k * self.t)
        for parameter in self.parameters:
            parameter['array'] -= parameter['grad'] * learn_rate
        self.t += 1


class Inverse_Decay:
    def __init__(self, parameters, learn_rate=1e-3, k=0.01):
        self.learn_rate = learn_rate
        self.t = 0
        self.parameters = parameters
        self.k = k

    def UpdateWeights(self):
        learn_rate = self.learn_rate / (1 + self.k * self.t)
        for parameter in self.parameters:
            parameter["array"] -= parameter["grad"] * learn_rate
        self.t += 1


class Momentum:
    def __init__(self, parameters, learn_rate=1e-4, mu=0.2):
        self.learn_rate = learn_rate
        self.mu = mu
        self.parameters = parameters
        self.v = [np.zeros(param['array'].shape) for param in self.parameters]

    def UpdateWeights(self):
        for i, param in enumerate(self.parameters):
            self.v[i] = self.mu * self.v[i] - self.learn_rate * param['grad']
            param['array'] += self.v[i]


class AdaGrad:
    # parameters is a dictionary [{array for w_0's: gradients for w_0's}, {array for w_1's: ...}]
    def __init__(self, parameters, learn_rate=1e-3, k=.9):
        self.k = k
        self.learn_rate = learn_rate
        self.parameters = parameters

        # same dims as parameters["grad"]
        self.gs = []

        for parameter_dictionary in parameters:
            array_shape = parameter_dictionary["array"].shape
            self.gs.append(np.zeros(array_shape))

    def UpdateWeights(self):
        counter = 0

        for parameter in self.parameters:
            self.gs[counter] = self.gs[counter] * self.k + parameter["grad"] ** 2
            # print(self.gs[counter])
            parameter["array"] = parameter["array"] - self.learn_rate / (np.sqrt(self.gs[counter]) + 1e-99) * parameter["grad"]

            counter += 1


class RMS_Prop:
    def __init__(self, parameters, learn_rate=1e-2, k=0.5):
        """

        :param parameters:
        :param learn_rate:
        :param k:
        """
        self.parameters = parameters
        self.learn_rate = learn_rate
        self.k = k
        self.g = [np.zeros(param['array'].shape) for param in parameters]

    def UpdateWeights(self):
        """

        :return:
        """
        for idx, param in enumerate(self.parameters):
            self.g[idx] = self.k * self.g[idx] + (1 - self.k) * param['grad']**2
            param['array'] -= param['grad'] * self.learn_rate/np.sqrt(self.g[idx] + 1e-99)


class Adam:
    def __init__(self, parameters, learn_rate=1e-2, k_1=0.5, k_2=0.5):
        self.parameters = parameters
        self.learn_rate = learn_rate
        self.k_1 = k_1
        self.k_2 = k_2
        self.m = [np.zeros(param['array'].shape) for param in parameters]
        self.v = [np.zeros(param['array'].shape) for param in parameters]
        self.t = 0

    def UpdateWeights(self):
        self.t += 1
        for idx, param in enumerate(self.parameters):
            self.m[idx] = self.k_1 * self.m[idx] + (1-self.k_1) * param['grad']
            self.v[idx] = self.k_2 * self.v[idx] + (1-self.k_2) * param['grad']**2
            m_hat = self.m[idx] / (1 - self.k_1**self.t)
            v_hat = self.v[idx] / (1 - self.k_2**self.t)
            param['array'] -= self.learn_rate * m_hat / np.sqrt(v_hat + 1e-99)
