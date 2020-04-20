import numpy as np

class GD:
    def __init__(self, parameters, learn_rate = 1e-3):
        self.learn_rate = learn_rate
        self.parameters = parameters

    def update_weights(self):
        for parameter in self.parameters:
            parameter['array'] -= parameter['grad'] * self.learn_rate