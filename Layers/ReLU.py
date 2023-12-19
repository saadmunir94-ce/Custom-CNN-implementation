import numpy as np


class ReLU:

    def __init__(self):
        self.x_input = None

    def forward(self, input_tensor):
        self.x_input = input_tensor
        input_tensor = input_tensor * (input_tensor > 0).astype(float)
        return input_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor * (self.x_input > 0)
        return error_tensor
