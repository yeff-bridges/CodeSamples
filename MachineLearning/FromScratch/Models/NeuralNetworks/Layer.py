import numpy as np

class Layer(BaseLayer):
    """
    Typical Feed-Forward, Full Connected Layer with Activation and Batch Normalization options.
    Requires that another BaseLayer object come after, so a separate type of Layer is used for output.
    """
    def __init__(self, size_in, size_out, activation_func=Sigmoid(), learn_rate=1e-3, batch_norm_p=1):
        super().__init__()
        self.w = {"array": np.random.randn(size_in, size_out) * np.sqrt(1 / (size_in + size_out))}
        self.b = {"array": np.random.randn(1, size_out) * np.sqrt(1 / (size_in + size_out))}
        self.activation_func = activation_func
        self.learn_rate = learn_rate
        self.batch_norm_p = batch_norm_p
        self.mu_h = 0
        self.sig_h = 1

    def forward(self, z_in):
        h = z_in @ self.w["array"] + self.b["array"]
        mu_h = np.mean(h, keepdims=True, axis=0)
        self.mu_h = self.mu_h * self.batch_norm_p + mu_h * (1 - self.batch_norm_p)
        sig_h = np.std(h, keepdims=True, axis=0)
        self.sig_h = self.sig_h * self.batch_norm_p + sig_h * (1 - self.batch_norm_p)
        h_norm = (h - self.mu_h) / self.sig_h
        z_out = self.activation_func(h_norm)
        return self.after.forward(z_out) if not self.after == None else z_out

    def backward(self, z_in, y):
        h = z_in @ self.w["array"] + self.b["array"]
        mu_h = np.mean(h, keepdims=True, axis=0)
        self.mu_h = self.mu_h * self.batch_norm_p + mu_h * (1 - self.batch_norm_p)
        sig_h = np.std(h, keepdims=True, axis=0)
        self.sig_h = self.sig_h * self.batch_norm_p + sig_h * (1 - self.batch_norm_p)
        h_norm = (h - self.mu_h) / self.sig_h
        z_out = self.activation_func(h_norm)

        grad_after = self.after.backward(z_out, y)
        grad_h_norm = grad_after * self.activation_func.grad(h_norm)
        grad_h = grad_h_norm / self.sig_h

        grad_w = z_in.T @ grad_h
        grad_b = np.sum(grad_h, axis=0)
        grad_z_in = grad_h @ self.w["array"].T

        self.w["grad"] = grad_w
        self.b["grad"] = grad_b

        return grad_z_in

    def predict(self, z_in):
        return self.after.predict(self.forward(z_in))

    def get_parameters(self):
        parameters = self.after.get_parameters()
        my_parameters = [self.w, self.b]
        parameters.extend(my_parameters)
        return parameters