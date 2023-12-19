import numpy as np
import math


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.__max_locations = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.no_channels = input_tensor.shape[1]
        self.input_spatial = input_tensor.shape[2:]
        self.output_y = math.floor((self.input_spatial[0] - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        self.output_x = math.floor((self.input_spatial[1] - self.pooling_shape[1]) / self.stride_shape[1]) + 1
        output_tensor = np.zeros((self.batch_size, self.no_channels, self.output_y, self.output_x))
        self.__max_locations = np.zeros(output_tensor.shape, dtype='int')
        for b in range(self.batch_size):
            for c in range(self.no_channels):
                for i in range(0, self.stride_shape[0] * self.output_y, self.stride_shape[0]):
                    for j in range(0, self.stride_shape[1] * self.output_x, self.stride_shape[1]):
                        pool = input_tensor[b, c, i:i + self.pooling_shape[0], j:j + self.pooling_shape[1]]
                        y = int(i / self.stride_shape[0])
                        x = int(j / self.stride_shape[1])
                        output_tensor[b, c, y, x] = np.amax(pool)
                        index = np.unravel_index(np.argmax(pool, axis=None), pool.shape)
                        # shifting to global co-ordinates
                        index = (index[0] + i, index[1] + j)
                        self.__max_locations[b, c, y, x] = index[0] * self.input_spatial[1] + index[1]

        return output_tensor

    def backward(self, error_tensor):
        prev_error = np.zeros((self.batch_size, self.no_channels, self.input_spatial[0], self.input_spatial[1]))
        for b in range(self.batch_size):
            for c in range(self.no_channels):
                for i in range(self.output_y):
                    for j in range(self.output_x):
                        index = np.unravel_index(self.__max_locations[b, c, i, j], self.input_spatial)
                        prev_error[b, c, index[0], index[1]] += error_tensor[b, c, i, j]

        return prev_error
print("Hello World")
