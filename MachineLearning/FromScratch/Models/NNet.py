from DeeperDream.Activation.Class import *
import numpy as np

class Base_Layer:
    """
    Superclass for all of the layers.

    Contains methods for connecting layers together.
    """
    def __init__(self):
        self.end = self
        self.after = None

    def AddAfter(self, layer):
        self.end.after = layer
        self.end = layer
        return self

    def __rshift__(self, layer):
        """
        Wrapper for self.AddAfter()

        Allows us to use the >> operator for constructing our models.

        Example - A simple two layer model with an input shape of 2 and an output shape of 4:
            Layer(2, 3) >> Layer(3, 4)

        :param layer: The next layer in the model; the current layer will feed into this
        :return: self
        """
        return self.AddAfter(layer)


class Layer(Base_Layer):
    """
    Typical Feed-Forward, Full Connected Layer with Activation and Batch Normalization options.
    Requires that another Base_Layer object come after, so a separate type of Layer is used for output.
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

    def Forward(self, z_in):
        h = z_in @ self.w["array"] + self.b["array"]
        mu_h = np.mean(h, keepdims=True, axis=0)
        self.mu_h = self.mu_h * self.batch_norm_p + mu_h * (1 - self.batch_norm_p)
        sig_h = np.std(h, keepdims=True, axis=0)
        self.sig_h = self.sig_h * self.batch_norm_p + sig_h * (1 - self.batch_norm_p)
        h_norm = (h - self.mu_h) / self.sig_h
        z_out = self.activation_func(h_norm)
        return z_out

    def Backward(self, z_in, y):
        h = z_in @ self.w["array"] + self.b["array"]
        mu_h = np.mean(h, keepdims=True, axis=0)
        self.mu_h = self.mu_h * self.batch_norm_p + mu_h * (1 - self.batch_norm_p)
        sig_h = np.std(h, keepdims=True, axis=0)
        self.sig_h = self.sig_h * self.batch_norm_p + sig_h * (1 - self.batch_norm_p)
        h_norm = (h - self.mu_h) / self.sig_h
        z_out = self.activation_func(h_norm)

        grad_after = self.after.Backward(z_out, y)
        grad_h_norm = grad_after * self.activation_func.D(h_norm)
        grad_h = grad_h_norm / self.sig_h

        grad_w = z_in.T @ grad_h
        grad_b = np.sum(grad_h, axis=0)
        grad_z_in = grad_h @ self.w["array"].T

        self.w["grad"] = grad_w
        self.b["grad"] = grad_b

        return grad_z_in

    def Predict(self, z_in):
        return self.after.Predict(self.Forward(z_in))

    def GetParameters(self):
        parameters = self.after.GetParameters()
        my_parameters = [self.w, self.b]
        parameters.extend(my_parameters)
        return parameters


class Output_Layer(Layer):
    """
    Similar to Layer, but this one doesn't assume it has as Base_Layer after it
    """
    def Backward(self, z_in, y):
        y_hat = self.Forward(z_in)
        grad_h = y_hat - y

        grad_w = z_in.T @ grad_h
        grad_b = np.sum(grad_h, axis=0)
        grad_z_in = grad_h @ self.w["array"].T

        self.w["grad"] = grad_w
        self.b["grad"] = grad_b

        return grad_z_in

    def Predict(self, z_in):
        return self.Forward(z_in)

    def GetParameters(self):
        my_parameters = [self.w, self.b]
        return my_parameters


class Split_Layer(Base_Layer):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

        self.left.AddAfter(Empty_layer(0))
        self.right.AddAfter(Empty_layer(0))

    def Forward(self, z_in):
        z_left = self.left.Forward(z_in)
        z_right = self.right.Forward(z_in)
        z_out = np.hstack([z_left, z_right])
        return z_out

    def Backward(self, z_in, y):
        z_left = self.left.Predict(z_in)
        z_right = self.right.Predict(z_in)
        z_out = np.hstack([z_left, z_right])

        grad_after = self.after.Backward(z_out, y)
        grad_left = grad_after[:, :z_left.shape[1]]
        grad_right = grad_after[:, z_left.shape[1]:]

        self.left.AddAfter(Empty_layer(grad_left))
        self.right.AddAfter(Empty_layer(grad_right))

        grad_z_left = self.left.Backward(z_in, y)
        grad_z_right = self.right.Backward(z_in, y)

        grad_z_in = grad_z_left + grad_z_right

        return grad_z_in

    def Predict(self, z_in):
        z_left = self.left.Predict(z_in)
        z_right = self.right.Predict(z_in)
        z_out = np.hstack([z_left, z_right])
        return self.after.Predict(z_out)

    def GetParameters(self):
        parameters = self.after.GetParameters()

        parameters_left = self.left.GetParameters()
        parameters_right = self.right.GetParameters()

        parameters.extend(parameters_left)
        parameters.extend(parameters_right)
        return parameters_left


class Bypass_Layer(Base_Layer):
    """
    A layer used to create a Bypass between Base_Layers
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.AddAfter(Empty_layer(0))

    def Forward(self, z_in):
        z_model = self.model.Forward(z_in)
        z_out = np.hstack([z_in, z_model])
        return z_out

    def Backward(self, z_in, y):
        z_model = self.model.Predict(z_in)
        z_out = np.hstack([z_in, z_model])

        grad_after = self.after.Backward(z_out, y)
        grad_z_left = grad_after[:, :z_in.shape[1]]
        grad_model = grad_after[:, z_in.shape[1]:]

        self.model.AddAfter(Empty_layer(grad_model))

        grad_z_model = self.model.Backward(z_in, y)

        grad_z_in = grad_z_left + grad_z_model

        return grad_z_in

    def Predict(self, z_in):
        z_model = self.model.Predict(z_in)
        z_out = np.hstack([z_in, z_model])
        return self.after.Predict(z_out)

    def GetParameters(self):
        parameters = self.after.GetParameters()
        parameters_model = self.model.GetParameters()

        parameters.extend(parameters_model)
        return parameters


class Empty_layer(Base_Layer):
    """
    A dummy Base_Layer utilized by the Bypass_Layer when connecting ends of the Bypass
    """
    def __init__(self, grad):
        super().__init__()
        self.grad = grad

    def Backward(self, *args, **kwargs):
        return self.grad

    @staticmethod
    def Forward(z_in):
        return z_in

    @staticmethod
    def Predict(z_in):
        return z_in

    @staticmethod
    def GetParameters():
        return []


class Dropout_Layer(Base_Layer):
    """
    A Base_Layer which randomly zeros a portion of the outputs of the Base_Layer coming before it.
    The shape of the input is preserved, and the random zeroing is only performed during training
    """
    def __init__(self, drop_rate=.5):
        super().__init__()
        self.drop_rate = drop_rate

    def Backward(self, z_in, y):
        mask = np.random.rand(np.shape(z_in)[0], np.shape(z_in)[1]) > self.drop_rate
        z_out = mask * z_in

        grad_after = self.after.Backward(z_out, y)
        grad_out = grad_after * mask

        return grad_out

    def Forward(self, z_in):
        mask = np.random.rand(np.shape(z_in)[0], np.shape(z_in)[1]) > self.drop_rate
        z_out = mask * z_in
        return z_out

    def Predict(self, z_in):
        return self.after.Predict(z_in)

    def GetParameters(self):
        parameters = self.after.GetParameters()
        return parameters


class Noise_Injection_Layer(Base_Layer):
    """
    A Base_Layer which adds random noise to the outputs of the Base_Layer fed into it
    The shape of the inputs is preserved, and the noise is only injected during training
    """
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def Forward(self, z_in):
        mu = np.mean(z_in, keepdims=True, axis=0)
        std = np.std(z_in, keepdims=True, axis=0)
        z_out = z_in + (np.random.randn(z_in.shape[0], z_in.shape[1]) * std + mu) * self.sigma
        return z_out

    def Backward(self, z_in, y):
        z_out = self.Forward(z_in)

        grad_out = self.after.Backward(z_out, y)
        return grad_out

    def Predict(self, z_in):
        return self.after.Predict(z_in)

    def GetParameters(self):
        parameters = self.after.GetParameters()
        return parameters


class Batch_Norm_Layer(Base_Layer):
    """
    A Base_Layer which normalizes the outputs of the previous Base_Layer based on a moving average of all outputs seen during training.
    This normalization preserves the input shape, and the moving average is only updated during training
    """
    def __init__(self, norm_p):
        super().__init__()
        self.mu = 0
        self.sig = 1
        self.norm_p = norm_p

    def Forward(self, z_in):
        mu = np.mean(z_in, keepdims=True, axis=0)
        self.mu = self.norm_p * self.mu + (1 - self.norm_p) * mu
        sig = np.std(z_in, keepdims=True, axis=0)
        self.sig = self.norm_p * self.sig + (1 - self.norm_p) * sig

        z_out = (z_in - self.mu) / (self.sig + 1e-99)
        return z_out

    def Backward(self, z_in, y):
        z_out = self.Forward(z_in)

        grad_after = self.after.Backward(z_out, y)
        grad_out = grad_after / self.sig
        return grad_out

    def Predict(self, z_in):
        mu = np.mean(z_in, keepdims=True, axis=0)
        mu = self.norm_p * self.mu + (1 - self.norm_p) * mu
        sig = np.std(z_in, keepdims=True, axis=0)
        sig = self.norm_p * self.sig + (1 - self.norm_p) * sig

        z_out = (z_in - mu) / (sig + 1e-99)
        return self.after.Predict(z_out)

    def GetParameters(self):
        parameters = self.after.GetParameters()
        return parameters
