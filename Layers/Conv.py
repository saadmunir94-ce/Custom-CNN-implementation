import numpy as np
import math
import sys
from scipy import signal
from Layers import Initializers
from Optimization import *
import copy


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.kernel_shape = list(self.convolution_shape)

        self.kernel_shape.insert(0, int(self.num_kernels))
        # self.kernel_shape = tuple(self.kernel_shape)

        self.weights = np.random.random_sample(self.kernel_shape)
        self.is_2D = True
        if len(self.convolution_shape) == 2:
            self.is_2D = False

        self.bias = np.random.rand(num_kernels)
        self.stride_y = 1
        self.stride_x = 1
        self.input_tensor = None
        self.is_Gradient = False
        self._final_weights_gradient = np.zeros(self.kernel_shape)
        self._final_bias_gradient = np.zeros(self.num_kernels)
        self._bias_optimizer = None
        self._weights_optimizer = None
        self.fan_in = np.prod(self.convolution_shape)
        self.fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])

        # print("Stride shape is ", stride_shape)
        if isinstance(stride_shape, tuple):
            self.stride_y = stride_shape[0]
            self.stride_x = stride_shape[1]
        else:

            self.stride_y = stride_shape[0]
            self.stride_x = stride_shape[0]

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        # tuple
        self.input_spatial = input_tensor.shape[2:]
        self.is_Gradient = False
        if self.is_2D:
            feature_map = np.zeros((input_tensor.shape[0], self.num_kernels,
                                    math.ceil(input_tensor.shape[2] / self.stride_y),
                                    math.ceil(input_tensor.shape[3] / self.stride_x)))
        else:
            feature_map = np.zeros(
                (input_tensor.shape[0], self.num_kernels, math.ceil(input_tensor.shape[2] / self.stride_y)))

        if self.is_2D:
            self.pad_y_before = math.ceil((self.convolution_shape[1] - 1) / 2)
            self.pad_y_after = math.floor((self.convolution_shape[1] - 1) / 2)
            self.pad_x_before = math.ceil((self.convolution_shape[2] - 1) / 2)
            self.pad_x_after = math.floor((self.convolution_shape[2] - 1) / 2)
        else:
            self.pad_y_before = math.ceil((self.convolution_shape[1] - 1) / 2)
            self.pad_y_after = math.floor((self.convolution_shape[1] - 1) / 2)

        if self.is_2D:
            self.padded_input_tensor = np.pad(input_tensor,
                                              [(0, 0), (0, 0), (self.pad_y_before, self.pad_y_after),
                                               (self.pad_x_before, self.pad_x_after)],
                                              mode='constant')

        else:
            self.padded_input_tensor = np.pad(input_tensor, [(0, 0), (0, 0), (self.pad_y_before, self.pad_y_after)],
                                              mode='constant')

        for i in range(len(input_tensor)):
            for j in range(int(self.num_kernels)):

                # correlation in FORWARD PASS
                channel = signal.correlate(self.padded_input_tensor[i], self.weights[j], mode='valid')
                if self.is_2D:
                    channel = channel.reshape(channel.shape[1], channel.shape[2])
                else:
                    channel = channel.reshape(channel.shape[1])

                if self.is_2D:
                    feature_map[i, j] = channel[::self.stride_y, ::self.stride_x]
                else:
                    feature_map[i, j] = channel[::self.stride_y]

                feature_map[i, j] = feature_map[i, j] + self.bias[j]

        return feature_map

    def backward(self, error_tensor):
        # batch_size = error_tensor.shape[0]
        # rearranged kernels
        back_weights = np.stack((self.weights[0:self.num_kernels]), axis=1)

        if self.is_2D:
            back_weights = np.flip(back_weights, axis=1)

        if self.is_2D:
            upsampled_error_tensor = np.zeros(
                (self.batch_size, self.num_kernels, self.input_spatial[0], self.input_spatial[1]))
            # adding strides
            upsampled_error_tensor[:, :, ::self.stride_y, ::self.stride_x] = error_tensor
            # same padding for same convolution
            upsampled_error_tensor = np.pad(upsampled_error_tensor,
                                            [(0, 0), (0, 0), (self.pad_y_before, self.pad_y_after),
                                             (self.pad_x_before, self.pad_x_after)], mode='constant')

            next_error_tensor = np.zeros(
                (self.batch_size, self.convolution_shape[0], self.input_spatial[0], self.input_spatial[1]))



        else:
            upsampled_error_tensor = np.zeros((self.batch_size, self.num_kernels, self.input_spatial[0]))
            # adding strides
            upsampled_error_tensor[:, :, ::self.stride_y] = error_tensor
            # same padding for same convolution
            upsampled_error_tensor = np.pad(upsampled_error_tensor,
                                            [(0, 0), (0, 0), (self.pad_y_before, self.pad_y_after)], mode='constant')

            next_error_tensor = np.zeros((self.batch_size, self.convolution_shape[0], self.input_spatial[0]))
        for i in range(self.batch_size):
            for j in range(len(back_weights)):

                # CONVOLUTION IN BACKWARD PASS
                # print("we do convolution")
                channel = signal.convolve(upsampled_error_tensor[i], back_weights[j], mode='valid')
                '''
                if self.is_2D:
                    channel = channel.reshape(channel.shape[1], channel.shape[2])
                else:
                    channel = channel.reshape(channel.shape[1])
                '''
                channel = channel.reshape(channel.shape[1:])

                next_error_tensor[i, j] = channel

        # Now we have the gradient with respect to the lower layers!

        # Paddings for correlation with channels of error_tensor
        '''
        if self.is_2D:
            pad_before_y = math.ceil((self.convolution_shape[1]-1)/2)
            pad_after_y  = math.floor((self.convolution_shape[1]-1)/2)
            pad_before_x  = math.ceil((self.convolution_shape[2]-1)/2)
            pad_after_x = = math.floor((self.convolution_shape[2]-1)/2)
        else:
            pad_before_y = math.ceil((self.convolution_shape[1]-1)/2)
            pad_after_y  = math.floor((self.convolution_shape[1]-1)/2)

        #padding for correlation with channels of error_tensor
        if self.is_2D:
            padded_input_tensor = np.pad(self.input_tensor,[(0,0),(0,0),(self.pad_y_before,self.pad_y_after),(self.pad_x_before,self.pad_x_after)],mode='constant')

        else:
            padded_input_tensor = np.pad(self.input_tensor,[(0,0),(0,0),(self.pad_y_before,self.pad_y_after)],mode='constant')
        if (self.padded_input_tensor == padded_input_tensor).all:
            print("Our assumption is true!! Hooray")
        else:
            print("Our ASSUMPTION= WRONG")
        '''

        w_gradient_shape = self.kernel_shape.copy()
        w_gradient_shape.insert(0, self.batch_size)
        self.gradient_tensor = np.zeros(w_gradient_shape)
        self.bias_gradient = np.zeros((self.batch_size, self.num_kernels))
        if self.is_2D:
            self.bias_gradient = np.sum(upsampled_error_tensor, axis=(2, 3))
        else:
            self.bias_gradient = np.sum(upsampled_error_tensor, axis=2)
        if self.is_2D:
            self.padded_input_tensor = np.pad(self.padded_input_tensor,
                                              [(0, 0), (0, 0), (self.pad_y_before, self.pad_y_after),
                                               (self.pad_x_before, self.pad_x_after)],
                                              mode='constant')
        else:
            self.padded_input_tensor = np.pad(self.padded_input_tensor,
                                              [(0, 0), (0, 0), (self.pad_y_before, self.pad_y_after),
                                               ],
                                              mode='constant')

        for i in range(self.batch_size):
            for j in range(self.num_kernels):
                kernel_filter = upsampled_error_tensor[i, j]

                '''
                if self.is_2D:
                    kernel_filter = kernel_filter.reshape(1, kernel_filter.shape[0], kernel_filter.shape[1])
                else:
                    kernel_filter = kernel_filter.reshape(1, kernel_filter.shape[0])
                '''
                new_shape = list(kernel_filter.shape)
                new_shape.insert(0, 1)
                kernel_filter = kernel_filter.reshape(new_shape)

                self.gradient_tensor[i, j] = signal.correlate(self.padded_input_tensor[i], kernel_filter, mode='valid')

                '''
                if self.is_2D:
                    for l in range(self.convolution_shape[0]):
                        # self.gradient_tensor[i, j, l] = signal.correlate(self.input_tensor[i, l],
                        #                                                kernel_filter,
                        #                                               mode='valid')
                        self.gradient_tensor[i, j, l] = signal.correlate(self.padded_input_tensor[i, l],
                                                                         kernel_filter,
                                                                         mode='valid')

                else:
                    kernel_filter = kernel_filter.reshape(1, kernel_filter.shape[0])

                    self.gradient_tensor[i, j] = signal.correlate(self.padded_input_tensor[i], kernel_filter,
                                                                  mode='valid')
                '''
        self._final_weights_gradient = np.sum(self.gradient_tensor, axis=0)
        self._final_bias_gradient = np.sum(self.bias_gradient, axis=0)
        self.is_Gradient = True
        # print("Final_Weight_Gradient has shape", self._final_weights_gradient.shape)
        # print("Final_Bias_Gradient has shape", self._final_bias_gradient.shape)
        # return next_error_tensor, self.gradient_tensor, self.bias_gradient
        if self._weights_optimizer is not None:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self._final_weights_gradient)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._final_bias_gradient)
        return next_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)

        return None

    @property
    def gradient_weights(self):
        if self.is_Gradient:
            return self._final_weights_gradient

    @property
    def gradient_bias(self):
        if self.is_Gradient:
            return self._final_bias_gradient

    @property
    def optimizer(self):
        return self._weights_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._weights_optimizer = copy.deepcopy(val)
        self._bias_optimizer = copy.deepcopy(val)

    print("Done")
