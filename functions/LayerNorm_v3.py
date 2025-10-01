import numpy as np
import torch

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
            # Initialize parameters for both numpy and torch
            self.weight_np = np.ones(self.normalized_shape, dtype=np.float32)
            self.bias_np = np.zeros(self.normalized_shape, dtype=np.float32) if bias else None

            self.weight_torch = torch.ones(self.normalized_shape)
            self.bias_torch = torch.zeros(self.normalized_shape) if bias else None
        else:
            self.weight_np, self.bias_np = None, None
            self.weight_torch, self.bias_torch = None, None

    def __call__(self, x):
        """
        Apply Layer Normalization.

        Args:
            x (np.ndarray or torch.Tensor): Input tensor of shape (..., *normalized_shape).

        Returns:
            np.ndarray or torch.Tensor: Normalized tensor with the same shape as input.
        """
        # Case 1: Torch tensor
        if isinstance(x, torch.Tensor):
            axes = tuple(range(-len(self.normalized_shape), 0))
            mean = x.mean(dim=axes, keepdim=True)
            var = x.var(dim=axes, keepdim=True, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)

            if self.elementwise_affine:
                shape = (1,) * (x_hat.ndim - len(self.normalized_shape)) + self.normalized_shape
                x_hat = x_hat * self.weight_torch.view(shape)
                if self.bias_torch is not None:
                    x_hat = x_hat + self.bias_torch.view(shape)

            return x_hat

        # Case 2: NumPy array
        elif isinstance(x, np.ndarray):
            axes = tuple(range(-len(self.normalized_shape), 0))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            x_hat = (x - mean) / np.sqrt(var + self.eps)

            if self.elementwise_affine:
                shape = (1,) * (x_hat.ndim - len(self.normalized_shape)) + self.normalized_shape
                x_hat = x_hat * self.weight_np.reshape(shape)
                if self.bias_np is not None:
                    x_hat = x_hat + self.bias_np.reshape(shape)

            return x_hat

        else:
            raise TypeError(f"Unsupported input type {type(x)}. Use np.ndarray or torch.Tensor.")
