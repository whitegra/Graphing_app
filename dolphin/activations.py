# -*- coding: utf-8 -*-
import numpy as np
from .tensor import Tensor

def softmax(x):
    """
    Implements softmax function: exp(x) / sum(exp(x))
    """
    exps = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))  # Stability trick
    out = Tensor(exps / np.sum(exps, axis=-1, keepdims=True), requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            x.grad = (out.grad - (out.grad * out.data).sum(axis=-1, keepdims=True)) * out.data

    out._backward = _backward
    out._prev = {x}
    return out

def gelu(x):
    """
    Implements GELU activation function used in Transformers.
    """
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data ** 3)))
    out = Tensor(x.data * cdf, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            x.grad = out.grad * (cdf + x.data * (1 - cdf))

    out._backward = _backward
    out._prev = {x}
    return out

