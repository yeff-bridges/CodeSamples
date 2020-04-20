import numpy as np

class BypassLayer(BaseLayer):
    """
    A layer used to create a Bypass between Base_Layers
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.add_after(EmptyLayer(0))

    def forward(self, z_in):
        z_model = self.model.forward(z_in)
        z_out = np.hstack([z_in, z_model])
        return z_out

    def backward(self, z_in, y):
        z_model = self.model.predict(z_in)
        z_out = np.hstack([z_in, z_model])

        grad_after = self.after.backward(z_out, y)
        grad_z_left = grad_after[:, :z_in.shape[1]]
        grad_model = grad_after[:, z_in.shape[1]:]

        self.model.add_after(EmptyLayer(grad_model))

        grad_z_model = self.model.backward(z_in, y)

        grad_z_in = grad_z_left + grad_z_model

        return grad_z_in

    def predict(self, z_in):
        z_model = self.model.predict(z_in)
        z_out = np.hstack([z_in, z_model])
        return self.after.predict(z_out)

    def get_parameters(self):
        parameters = self.after.get_parameters()
        parameters_model = self.model.get_parameters()

        parameters.extend(parameters_model)
        return parameters