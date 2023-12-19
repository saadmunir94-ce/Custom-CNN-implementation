import numpy as np
import math


class Constant:
    def __init__(self, c=0.1):
        self.constant = c

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant)


class UniformRandom:

    def initialize(self, weights_shape, fan_in, fan_out):
        #return np.random.rand(weights_shape[0], weights_shape[1])
        return np.random.random_sample(weights_shape)


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0.0, sigma, weights_shape)


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, sigma, weights_shape)
