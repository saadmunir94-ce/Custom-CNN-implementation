import numpy as np


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape

        return input_tensor.reshape(self.input_shape[0], np.prod(self.input_shape[1:]) )

    def backward(self, error_tensor):

        return error_tensor.reshape(self.input_shape)
