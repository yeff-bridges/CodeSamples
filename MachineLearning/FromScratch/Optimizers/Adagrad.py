import numpy as np

class Adagrad:
    def __init__(self, parameters, learn_rate=1e-3, k=.9):
        self.k = k
        self.learn_rate = learn_rate
        self.parameters = parameters

        self.gs = []

        for parameter_dictionary in parameters:
            array_shape = parameter_dictionary["array"].shape
            self.gs.append(np.zeros(array_shape))

    def update_weights(self):
        counter = 0

        for parameter in self.parameters:
            self.gs[counter] = self.gs[counter] * self.k + parameter["grad"] ** 2
            parameter["array"] = parameter["array"] - self.learn_rate / (np.sqrt(self.gs[counter]) + 1e-99) * parameter["grad"]

            counter += 1