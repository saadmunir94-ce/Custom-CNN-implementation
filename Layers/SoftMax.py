import numpy as np


class SoftMax:

    def __init__(self):
        self.class_output = None

    def forward(self, input_tensor):
        # find max along each row
        max_tensor = np.amax(input_tensor, axis=1).reshape(-1, 1)
        input_tensor = input_tensor - max_tensor
        input_tensor = np.exp(input_tensor)
        # sum along each row
        sum_tensor = np.sum(input_tensor, axis=1).reshape(-1, 1)
        self.class_output = np.divide(input_tensor, sum_tensor)
        return self.class_output.copy()

    def backward(self, error_tensor):
        return np.multiply(self.class_output,
                           error_tensor - np.einsum('...j,...j', error_tensor, self.class_output).reshape(-1, 1))


# Testing SoftMax
'''
input_tensor = np.array([
    [-3.44, 1.16, -0.81, 3.91],
    [-100, 10, 5, 3],
    [500, 10, 20, 50]

])

np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format})
soft_layer = SoftMax()
soft_output = soft_layer.forward(input_tensor)
print(soft_output)
error_tensor = np.array([

    [10, 5, 6, 1],
    [3, 4, 2, -1],
    [2, 1, 6, 8]
])
next_error = soft_layer.backward(error_tensor)
print("Error tensor for next layer is ",next_error)
'''