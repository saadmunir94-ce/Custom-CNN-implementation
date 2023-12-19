from __future__ import division
import numpy as np
import math
class CrossEntropyLoss:
   
    def __init__(self):
        self.y_output= None
        
    def forward(self, input_tensor, label_tensor):
        self.y_output = input_tensor
        eps = np.finfo(float).eps
        input_tensor = input_tensor+eps
        self.loss = np.sum(-np.log(input_tensor)*label_tensor).astype(float)
        return self.loss
        
    def backward(self, label_tensor):
        label_tensor = -(label_tensor/self.y_output).astype(float)
        return label_tensor
