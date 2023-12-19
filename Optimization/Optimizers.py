import numpy as np


class Sgd:
    def __init__(self, eta):
        # learning_rate
        self.l_rate = eta

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.l_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum:

    def __init__(self, l_rate, m_rate):
        self.l_rate = l_rate
        self.m_rate = m_rate
        self.past_gradient = 0.0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.past_gradient = self.m_rate * self.past_gradient - self.l_rate * gradient_tensor
        updated_weights = weight_tensor + self.past_gradient
        return updated_weights


class Adam:
    def __init__(self, l_rate, mu, rho):
        self.l_rate = l_rate
        self.mu = mu
        self.rho = rho
        self.past_gradient = 0
        self.r = 0
        self.index = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.index = self.index + 1
        self.past_gradient = self.mu * self.past_gradient + np.multiply(1 - self.mu, gradient_tensor)
        self.r = self.rho * self.r + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)

        # bias_Correction
        bias_free_v = np.divide(self.past_gradient, 1 - np.power(self.mu, self.index))
        bias_free_r = np.divide(self.r, 1 - np.power(self.rho, self.index))

        eps = np.finfo(float).eps
        updated_weights = weight_tensor - self.l_rate * np.divide(bias_free_v + eps, np.sqrt(bias_free_r) + eps)
        return updated_weights
