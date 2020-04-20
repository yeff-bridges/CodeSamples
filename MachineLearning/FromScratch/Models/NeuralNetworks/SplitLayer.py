import numpy as np

class SplitLayer(BaseLayer):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

        self.left.add_after(EmptyLayer(0))
        self.right.add_after(EmptyLayer(0))

    def forward(self, z_in):
        z_left = self.left.forward(z_in)
        z_right = self.right.forward(z_in)
        z_out = np.hstack([z_left, z_right])
        return z_out

    def backward(self, z_in, y):
        z_left = self.left.predict(z_in)
        z_right = self.right.predict(z_in)
        z_out = np.hstack([z_left, z_right])

        grad_after = self.after.backward(z_out, y)
        grad_left = grad_after[:, :z_left.shape[1]]
        grad_right = grad_after[:, z_left.shape[1]:]

        self.left.add_after(EmptyLayer(grad_left))
        self.right.add_after(EmptyLayer(grad_right))

        grad_z_left = self.left.backward(z_in, y)
        grad_z_right = self.right.backward(z_in, y)

        grad_z_in = grad_z_left + grad_z_right

        return grad_z_in

    def predict(self, z_in):
        z_left = self.left.predict(z_in)
        z_right = self.right.predict(z_in)
        z_out = np.hstack([z_left, z_right])
        return self.after.predict(z_out)

    def get_parameters(self):
        parameters = self.after.get_parameters()

        parameters_left = self.left.get_parameters()
        parameters_right = self.right.get_parameters()

        parameters.extend(parameters_left)
        parameters.extend(parameters_right)
        return parameters_left