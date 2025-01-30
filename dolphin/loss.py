# -*- coding: utf-8 -*-
import numpy as np
from .tensor import Tensor
from .activations import softmax 

def cross_entropy_loss(logits, targets):
    """
    Implements numerically stable Cross-Entropy Loss.
    """
    probs = softmax(logits)  # Apply softmax before log
    clipped_probs = np.clip(probs.data, 1e-9, 1)  # ✅ Avoid log(0) errors
    
    N = targets.data.shape[0]
    #clipped_targets = np.clip(targets.data, 0, clipped_probs.shape[1] - 1)  # ✅ Fix out-of-bounds index
    #loss = -np.sum(np.log(clipped_probs[np.arange(N), clipped_targets])) / N
    
    clipped_targets = np.clip(targets.data, 0, clipped_probs.shape[1] - 1)  # ✅ Prevents out-of-bounds indexing
    loss = -np.sum(np.log(clipped_probs[np.arange(N), clipped_targets] + 1e-9)) / N  # ✅ Adds `1e-9` to prevent log(0)


    return Tensor(loss, requires_grad=True)  # ✅ Returns a Tensor for backpropagation
