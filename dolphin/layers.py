# -*- coding: utf-8 -*-
import numpy as np
from .tensor import Tensor  # Import our custom Tensor class

class Linear:
    def __init__(self, in_features, out_features):
        """
        Implements a fully connected layer: y = xW + b

        Args:
            in_features (int): Number of input neurons.
            out_features (int): Number of output neurons.
        """
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        """
        Forward pass: xW + b
        """
        return x.matmul(self.W) + self.b

class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        """
        Implements Layer Normalization.
        
        Args:
            embed_dim (int): Number of features to normalize.
            eps (float): Small value to prevent division by zero.
        """
        self.gamma = Tensor(np.ones(embed_dim), requires_grad=True)  # Learnable scale
        self.beta = Tensor(np.zeros(embed_dim), requires_grad=True)  # Learnable shift
        self.eps = eps

    def __call__(self, x):
        """
        Forward pass of LayerNorm.
        """
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        norm_x = (x.data - mean) / np.sqrt(var + self.eps)
        out = Tensor(self.gamma.data * norm_x + self.beta.data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                N = x.data.shape[-1]
                x.grad = out.grad * self.gamma.data
                x.grad -= np.sum(out.grad * norm_x, axis=-1, keepdims=True) * self.gamma.data / N
                x.grad /= np.sqrt(var + self.eps)

        out._backward = _backward
        out._prev = {x}
        return out



