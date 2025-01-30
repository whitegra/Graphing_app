import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        Initialize the custom Tensor object.
        
        Args:
            data (array-like): NumPy array or list.
            requires_grad (bool): Whether to track gradients.
        """
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None  # Gradient storage
        self._backward = lambda: None  # Backward function
        self._prev = set()  # Track previous tensors in computation graph
        
    def transpose(self, axes=None):
        """
        Returns a transposed version of the tensor.
        
        Args:
            axes (tuple, optional): Dimensions to swap. Defaults to None (reverses all dimensions).
        
        Returns:
            Tensor: A new tensor with transposed dimensions.
        """
        transposed_data = np.transpose(self.data, axes)  # Use NumPy's transpose
        return Tensor(transposed_data, requires_grad=self.requires_grad)
    
    def __truediv__(self, other):
        """
        Implements division for Tensor objects.
        Supports division by scalars (floats or ints).
        """
        if isinstance(other, (int, float)):  # Only allow division by scalars
            out = Tensor(self.data / other, requires_grad=self.requires_grad)
    
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + (out.grad / other)
            
            out._backward = _backward
            out._prev = {self}
            return out
        else:
            raise TypeError("Only scalar division (by int or float) is supported for Tensors.")


    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self):
        """
        Computes gradients using backpropagation.
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # Seed gradient for scalar outputs
        
        # Topological sort for backpropagation
        visited = set()
        topo_order = []
        def build_graph(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_graph(parent)
                topo_order.append(tensor)

        build_graph(self)

        # Backward pass
        for tensor in reversed(topo_order):
            tensor._backward()

    # ----------- Basic Operations -------------
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + out.grad
        
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad * other.data
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + out.grad * self.data

        out._backward = _backward
        out._prev = {self, other}
        return out

    def matmul(self, other):
        """
        Implements matrix multiplication (dot product)
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self.data.T @ out.grad
        
        out._backward = _backward
        out._prev = {self, other}
        return out

    def relu(self):
        """
        Implements ReLU activation function (back grad)
        """
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data))
                self.grad[self.data > 0] += out.grad[self.data > 0]
        
        out._backward = _backward
        out._prev = {self}
        return out
