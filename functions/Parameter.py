import numpy as np

class Parameter:
    """
    Manual equivalent of torch.nn.Parameter. 
    It is a wrapper around a NumPy array indicating that it is a learnable weight.
    """
    def __init__(self, data, requires_grad=True):
        """
        Args:
            data (np.ndarray or list/tuple): The weight data.
            requires_grad (bool): Indicates if the parameter should be updated during backprop.
        """
        # Ensure data is a NumPy array
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        
        # In a full system, you would store .grad here
        self.grad = None 
        
    def __repr__(self):
        # Custom representation to make it clear what it is
        grad_str = "requires_grad=True" if self.requires_grad else "requires_grad=False"
        return f"Parameter(shape={self.data.shape}, {grad_str})"

    def __array__(self, dtype=None):
        # Allows the object to be implicitly converted to a NumPy array in operations
        return self.data.astype(dtype) if dtype is not None else self.data
    
# Example usage:
# W = Parameter(np.random.randn(3, 4) * 0.01)
# print(W)
# # W can be used directly in NumPy operations, e.g.,
# # result = np.dot(np.zeros((2, 3)), W)
