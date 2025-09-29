############################################################################################################################################################
# Layer Normalization Implementation in NumPy
#
# Description:
#   This class implements Layer Normalization (LayerNorm) from scratch using NumPy. 
#   It normalizes the input tensor across the last dimensions (specified by normalized_shape),
#   ensuring zero mean and unit variance, with optional scale (weight) and shift (bias).
#
# Reference:
#   Layer Normalization: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py
#
# Author: Omar Mongy
# Date: 2025-09-29
############################################################################################################################################################

import numpy as np

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Args:
            normalized_shape (int or tuple): Shape of the input features to normalize over.
                                             Can be an integer (for 1D features) or a tuple (for multi-dim).
            eps (float): Small constant to avoid division by zero (numerical stability).
        """
        # Convert single int into tuple for consistency
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def __call__(self, x, weight=None, bias=None):
        """
        Apply Layer Normalization.

        Args:
            x (np.ndarray): Input tensor of shape (..., *normalized_shape).
            weight (np.ndarray): Scale parameter (gamma), must match normalized_shape. Default: None.
            bias (np.ndarray): Shift parameter (beta), must match normalized_shape. Default: None.

        Returns:
            np.ndarray: Normalized tensor with the same shape as input.
        """
        # Normalize along the last `len(normalized_shape)` axes
        axes = tuple(range(-len(self.normalized_shape), 0))

        # Compute mean and variance along the specified axes
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)

        # Normalize input: subtract mean, divide by std
        x_hat = (x - mean) / np.sqrt(var + self.eps)

        # Apply scale (gamma) if provided
        if weight is not None:
            # Reshape weight to broadcast correctly
            x_hat = x_hat * weight.reshape((1,) * (x_hat.ndim - len(self.normalized_shape)) + self.normalized_shape)

        # Apply shift (beta) if provided
        if bias is not None:
            # Reshape bias to broadcast correctly
            x_hat = x_hat + bias.reshape((1,) * (x_hat.ndim - len(self.normalized_shape)) + self.normalized_shape)

        return x_hat

############################################################################### ***** #############################################################################
############################################################################### usage #############################################################################
############################################################################### ***** #############################################################################
"""
# Input: [batch=2, seq_len=3, features=4]
x = np.random.randn(2, 3, 4).astype(np.float32)

# Pretrained weights (gamma & beta)
gamma = np.ones(4, dtype=np.float32) * 0.5   # scale
beta  = np.ones(4, dtype=np.float32) * 0.1   # shift

# Create LayerNorm object
layernorm = LayerNorm(4)

# Apply normalization
y = layernorm(x, weight=gamma, bias=beta)

# ---------------------------
# Print everything in detail
# ---------------------------

print("=== INPUT Feature Map (x) ===")
print(x)
print("Shape:", x.shape)

print("\n=== GAMMA (scale) ===")
print(gamma)

print("\n=== BETA (shift) ===")
print(beta)

# Compute mean/var per token (last dim = features)
mean = x.mean(axis=-1, keepdims=True)
var = x.var(axis=-1, keepdims=True)

print("\n=== MEAN Feature Map (per token) ===")
print(mean)
print("Shape:", mean.shape)

print("\n=== VARIANCE Feature Map (per token) ===")
print(var)
print("Shape:", var.shape)

normed = (x - mean) / np.sqrt(var + 1e-5)
print("\n=== NORMALIZED Feature Map (before gamma/beta) ===")
print(normed)
print("Shape:", normed.shape)

print("\n=== OUTPUT Feature Map (after gamma/beta) ===")
print(y)
print("Shape:", y.shape)

# ---------------------------
# Step-by-step check for one token
# ---------------------------
b, t = 0, 0  # batch=0, token=0
print("\n--- Detailed example for token [0,0,:] ---")
print("Input features :", x[b, t])
print("Mean           :", mean[b, t, 0])
print("Variance       :", var[b, t, 0])
print("Normalized     :", normed[b, t])
print("Output         :", y[b, t])
"""
############################################################################### ***** #############################################################################
############################################################################### _out_ #############################################################################
############################################################################### ***** #############################################################################
"""
=== INPUT Feature Map (x) ===
[[[ 1.1901388   0.11255071 -2.2968328   0.6632075 ]
  [ 1.4353874   0.31862846 -0.40012118  1.2791066 ]
  [-0.7469549  -0.20276845  0.9463874   1.5781711 ]]

 [[ 0.24581857 -2.010922   -0.94042253  1.4340802 ]
  [-0.42300263  0.10431881  2.3830469   1.1468687 ]
  [-1.5946132  -1.6923208   0.12389289  1.8670369 ]]]
Shape: (2, 3, 4)

=== GAMMA (scale) ===
[0.5 0.5 0.5 0.5]

=== BETA (shift) ===
[0.1 0.1 0.1 0.1]

=== MEAN Feature Map (per token) ===
[[[-0.08273394]
  [ 0.65825033]
  [ 0.3937088 ]]

 [[-0.31786144]
  [ 0.8028079 ]
  [-0.32400104]]]
Shape: (2, 3, 1)

=== VARIANCE Feature Map (per token) ===
[[[1.7792509 ]
  [0.55622447]
  [0.8413259 ]]

 [[1.6602678 ]
  [1.1515079 ]
  [2.1220026 ]]]
Shape: (2, 3, 1)

=== NORMALIZED Feature Map (before gamma/beta) ===
[[[ 0.95425665  0.14640245 -1.659882    0.5592228 ]
  [ 1.0420022  -0.4553724  -1.4190876   0.8324576 ]
  [-1.2435777  -0.6502932   0.6025428   1.2913278 ]]

 [[ 0.4374639  -1.3139598  -0.48316067  1.3596567 ]
  [-1.1423205  -0.650915    1.4726088   0.3206268 ]
  [-0.872246   -0.9393201   0.3074689   1.5040972 ]]]
Shape: (2, 3, 4)

=== OUTPUT Feature Map (after gamma/beta) ===
[[[ 0.57712835  0.17320123 -0.72994095  0.3796114 ]
  [ 0.6210011  -0.1276862  -0.6095438   0.5162288 ]
  [-0.52178884 -0.22514659  0.4012714   0.74566394]]

 [[ 0.31873196 -0.5569799  -0.14158034  0.77982837]
  [-0.47116026 -0.22545752  0.8363044   0.2603134 ]
  [-0.33612302 -0.36966005  0.25373444  0.85204864]]]
Shape: (2, 3, 4)

--- Detailed example for token [0,0,:] ---
Input features : [ 1.1901388   0.11255071 -2.2968328   0.6632075 ]
Mean           : -0.082733944
Variance       : 1.7792509
Normalized     : [ 0.95425665  0.14640245 -1.659882    0.5592228 ]
Output         : [ 0.57712835  0.17320123 -0.72994095  0.3796114 ]
"""
