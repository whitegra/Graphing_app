# -*- coding: utf-8 -*-
import numpy as np 

class SGD:
    def __init__(self, parameters, lr=0.01):
        """
        Implements Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters (list): List of Tensors (model weights).
            lr (float): Learning rate.
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Update parameters using SGD.
        """
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        """
        Reset gradients to zero.
        """
        for param in self.parameters:
            param.grad = None

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Implements Adam optimizer.
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {p: np.zeros_like(p.data) for p in parameters}
        self.v = {p: np.zeros_like(p.data) for p in parameters}
        self.t = 0

    def step(self):
        """
        Update parameters using Adam.
        """
        self.t += 1
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
                self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad ** 2)

                m_hat = self.m[p] / (1 - self.beta1 ** self.t)
                v_hat = self.v[p] / (1 - self.beta2 ** self.t)

                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Reset gradients to zero.
        """
        for p in self.parameters:
            p.grad = None


