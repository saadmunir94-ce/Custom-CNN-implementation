import numpy as np
from Optimization import Optimizers
from Layers import Initializers


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    def forward(self, input_tensor):

        self.input_tensor = np.append(input_tensor, np.ones((input_tensor.shape[0], 1)), axis=1)
        # isGTensor determines whether Gradient with respect to weights calculated or not
        self.IsGTensor = False
        # output_tensor = y_hats
        self.output_tensor = np.matmul(self.input_tensor, self.weights)
        return self.output_tensor

    def backward(self, error_tensor):
        next_error = np.matmul(error_tensor, self.weights.T)
        # gradient with respect to weights
        self._gradient_tensor = np.matmul(self.input_tensor.T, error_tensor)
        self.IsGTensor = True
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_tensor)

        return next_error[:, :self.input_size]

    @property
    def gradient_weights(self):
        if self.IsGTensor:
            return self._gradient_tensor

    def initialize(self, weights_initializer, bias_initializer):
        weight_shape = (self.input_size, self.output_size)
        bias_shape = (1, self.output_size)

        self.weights[:self.input_size, :] = weights_initializer.initialize(weight_shape, self.input_size,
                                                                           self.output_size)
        self.weights[self.input_size, :] = bias_initializer.initialize(bias_shape, 1, self.output_size)

        return None
