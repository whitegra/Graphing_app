# -*- coding: utf-8 -*-
import numpy as np
from .tensor import Tensor
from .activations import softmax
from .layers import LayerNorm
from .activations import gelu

class MultiHeadSelfAttention:
    def __init__(self, embed_dim, num_heads):
        """
        Implements Multi-Head Self-Attention.
        """
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = Tensor(np.random.randn(embed_dim, embed_dim) * 0.01, requires_grad=True)
        self.W_k = Tensor(np.random.randn(embed_dim, embed_dim) * 0.01, requires_grad=True)
        self.W_v = Tensor(np.random.randn(embed_dim, embed_dim) * 0.01, requires_grad=True)
        self.W_o = Tensor(np.random.randn(embed_dim, embed_dim) * 0.01, requires_grad=True)

    def __call__(self, x):
        """
        Forward pass of Multi-Head Self-Attention.
        """
        Q = x.matmul(self.W_q)
        K = x.matmul(self.W_k)
        V = x.matmul(self.W_v)

        #scores = Q.matmul(K.transpose()) / np.sqrt(self.head_dim)
        if len(K.data.shape) == 3:
           K = K.transpose((0, 2, 1))  # Only apply if K is 3D
        else:
            K = K.transpose()  # Default transpose for 2D tensors
        
        scores = Q.matmul(K) / np.sqrt(self.head_dim)  # Now K has correct shape

        #scores = Q.matmul(K.transpose((0, 2, 1))) / np.sqrt(self.head_dim)
        attn_weights = softmax(scores)
        out = attn_weights.matmul(V)
        return out.matmul(self.W_o)
    
class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        """
        Implements the Feed Forward Network (FFN).
        
        Args:
            embed_dim (int): Input embedding size.
            hidden_dim (int): Hidden layer size.
        """
        self.W1 = Tensor(np.random.randn(embed_dim, hidden_dim) * 0.01, requires_grad=True)
        self.b1 = Tensor(np.zeros(hidden_dim), requires_grad=True)
        self.W2 = Tensor(np.random.randn(hidden_dim, embed_dim) * 0.01, requires_grad=True)
        self.b2 = Tensor(np.zeros(embed_dim), requires_grad=True)

    def __call__(self, x):
        """
        Forward pass: GELU(xW1 + b1) W2 + b2
        """
        return gelu(x.matmul(self.W1) + self.b1).matmul(self.W2) + self.b2


class TransformerEncoderBlock:
    def __init__(self, embed_dim, num_heads, hidden_dim):
        """
        Implements a single Transformer Encoder Block.

        Args:
            embed_dim (int): Embedding size.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden size of the feedforward network.
        """
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.norm2 = LayerNorm(embed_dim)

    def __call__(self, x):
        """
        Forward pass through the Transformer Encoder Block.
        """
        # Multi-Head Self-Attention + Residual Connection + LayerNorm
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)  # Residual Connection

        # Feedforward Network + Residual Connection + LayerNorm
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)  # Residual Connection
        return out


