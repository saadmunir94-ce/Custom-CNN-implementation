import numpy as np
from Layers import *
from Optimization import *
import copy


class NeuralNetwork:
    def __init__(self, optimizer, w_initializer, b_initializer):
        self.optimizer = optimizer
        self.weights_initializer = w_initializer
        self.bias_initializer = b_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def append_trainable_layer(self, layer):
        # appends incoming layer
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        self.batch = self.data_layer.forward()
        input_tensor = self.batch[0]
        # self.batch[1] represents corresponding labels
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)

        loss = self.loss_layer.forward(input_tensor, self.batch[1])
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.batch[1])
        for i in range(len(self.layers) - 1, -1, -1):
            error_tensor = self.layers[i].backward(error_tensor)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):

        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)

        # output from SoftMax
        return input_tensor
