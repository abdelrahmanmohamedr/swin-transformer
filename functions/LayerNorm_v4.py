import numpy as np

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        """
        Args:
            normalized_shape (int or tuple): Shape of the input features to normalize over.
            eps (float): Small constant to avoid division by zero.
            elementwise_affine (bool): If True, learnable affine parameters (weight & bias) are created.
            bias (bool): If True and elementwise_affine=True, a bias parameter is created.
        """
        # Convert single int into tuple for consistency
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # Initialize learnable parameters
            self.weight = np.ones(self.normalized_shape, dtype=np.float32)
            self.bias = np.zeros(self.normalized_shape, dtype=np.float32) if bias else None
        else:
            self.weight, self.bias = None, None

    def __call__(self, x):
        """
        Apply Layer Normalization.

        Args:
            x (np.ndarray): Input tensor of shape (..., *normalized_shape).

        Returns:
            np.ndarray: Normalized tensor with the same shape as input.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Unsupported input type {type(x)}. Use np.ndarray.")

        axes = tuple(range(-len(self.normalized_shape), 0))
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)

        if self.elementwise_affine:
            shape = (1,) * (x_hat.ndim - len(self.normalized_shape)) + self.normalized_shape
            x_hat = x_hat * self.weight.reshape(shape)
            if self.bias is not None:
                x_hat = x_hat + self.bias.reshape(shape)

        return x_hat
