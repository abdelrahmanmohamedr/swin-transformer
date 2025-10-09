from typing import Optional
import torch
from torch import Tensor


class BatchNorm1dInference:
    """
    Inference-only BatchNorm1d (manual implementation).

    Performs:
        y = weight * (x - running_mean) / sqrt(running_var + eps) + bias

    Accepts:
      - input: Tensor of shape (N, C) or (N, C, L)
      - weight: Optional[Tensor] of shape (C,)  (gamma). If None, no scaling.
      - bias: Optional[Tensor] of shape (C,)    (beta). If None, no shift.
      - running_mean: Tensor of shape (C,)      (must be provided for inference)
      - running_var: Tensor of shape (C,)       (must be provided for inference)
      - eps: float numeric epsilon for numerical stability

    Notes:
      - This implementation uses the provided running_mean and running_var.
        If either running_mean or running_var is None, it falls back to computing
        batch statistics from the input (not recommended for strict inference).
      - Always ensure input and parameter tensors are on the same device/dtype,
        or the function will move/cast parameters to input's device/dtype.
    """

    def __init__(
        self,
        num_features: int,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        eps: float = 1e-5,
    ):
        self.num_features = num_features
        self.eps = float(eps)

        # store parameters (can be CPU tensors; we will move/cast them at forward time)
        self.weight = None if weight is None else weight.clone().detach()
        self.bias = None if bias is None else bias.clone().detach()
        self.running_mean = None if running_mean is None else running_mean.clone().detach()
        self.running_var = None if running_var is None else running_var.clone().detach()

        # basic shape checks (only if tensors provided)
        if self.weight is not None and self.weight.shape != (num_features,):
            raise ValueError(f"weight must have shape ({num_features},), got {self.weight.shape}")
        if self.bias is not None and self.bias.shape != (num_features,):
            raise ValueError(f"bias must have shape ({num_features},), got {self.bias.shape}")
        if self.running_mean is not None and self.running_mean.shape != (num_features,):
            raise ValueError(f"running_mean must have shape ({num_features},), got {self.running_mean.shape}")
        if self.running_var is not None and self.running_var.shape != (num_features,):
            raise ValueError(f"running_var must have shape ({num_features},), got {self.running_var.shape}")

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply batchnorm to `input` (inference).

        Returns a new tensor (does not mutate parameters).
        """
        if input.dim() not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")

        N, C = input.shape[0], input.shape[1]
        if C != self.num_features:
            raise ValueError(f"Expected input with C={self.num_features} channels, got C={C}")

        device = input.device
        dtype = input.dtype

        # Ensure parameter tensors are on same device/dtype as input
        if self.running_mean is not None:
            running_mean = self.running_mean.to(device=device, dtype=dtype)
        else:
            running_mean = None

        if self.running_var is not None:
            running_var = self.running_var.to(device=device, dtype=dtype)
        else:
            running_var = None

        weight = None if self.weight is None else self.weight.to(device=device, dtype=dtype)
        bias = None if self.bias is None else self.bias.to(device=device, dtype=dtype)

        # If running stats are missing (None), compute batch stats from input.
        # This is a fallback — in typical inference you must provide running_mean/var.
        if running_mean is None or running_var is None:
            # Compute mean/var across N and optionally L: axes = (0, 2) if 3D else (0,)
            if input.dim() == 3:
                # input shape (N, C, L)
                dims = (0, 2)
            else:
                # input shape (N, C)
                dims = (0,)
            batch_mean = input.mean(dim=dims)
            # Use unbiased=False to match training-time bias used for normalization in PyTorch forward
            batch_var = input.var(dim=dims, unbiased=False)
            # Use batch stats
            running_mean = batch_mean
            running_var = batch_var

        # Prepare shapes for broadcasting: make shape (1, C, 1) for 3D, (1, C) for 2D
        if input.dim() == 3:
            # shape (N, C, L)
            rm = running_mean.view(1, C, 1)
            rv = running_var.view(1, C, 1)
            if weight is not None:
                w = weight.view(1, C, 1)
            else:
                w = None
            if bias is not None:
                b = bias.view(1, C, 1)
            else:
                b = None
        else:
            # shape (N, C)
            rm = running_mean.view(1, C)
            rv = running_var.view(1, C)
            w = None if weight is None else weight.view(1, C)
            b = None if bias is None else bias.view(1, C)

        # compute inv std: 1 / sqrt(var + eps)
        invstd = (rv + self.eps).sqrt().reciprocal()

        # normalized = (x - mean) * invstd
        normalized = (input - rm) * invstd

        # apply affine if present: y = gamma * normalized + beta
        if w is not None:
            out = normalized * w
        else:
            out = normalized
        if b is not None:
            out = out + b

        return out


# ---------------------------
# Example usage (no execution here)
# ---------------------------
# Suppose you have a pretrained torch.nn.BatchNorm1d layer "bn" (already loaded)
# and an input tensor x, you can create the custom inference module as:
#
# custom = BatchNorm1dInference(
#     num_features=bn.num_features,
#     weight=bn.weight.detach() if bn.affine else None,
#     bias=bn.bias.detach() if bn.affine else None,
#     running_mean=bn.running_mean.detach() if bn.track_running_stats else None,
#     running_var=bn.running_var.detach() if bn.track_running_stats else None,
#     eps=bn.eps,
# )
#
# y_custom = custom.forward(x)
# # For verification you can compare with PyTorch builtin in eval mode:
# bn.eval()
# y_torch = bn(x)
# torch.allclose(y_custom, y_torch, atol=1e-6)
###################################################################accuracy###################################################
"""
import torch
import torch.nn as nn

# Assuming BatchNorm1dInference class is already defined/imported


def test_batchnorm1d_inference(batch_size=80, num_features=160, length=None, eps=1e-5):
    """
    Compare PyTorch BatchNorm1d (eval mode) vs. custom BatchNorm1dInference.
    """

    # Create input: either (N, C) or (N, C, L)
    if length is None:
        x = torch.randn(batch_size, num_features)
    else:
        x = torch.randn(batch_size, num_features, length)

    # Create pretrained BatchNorm1d and switch to eval (use running stats)
    bn = nn.BatchNorm1d(num_features, eps=eps)
    bn.eval()

    # Create custom inference BatchNorm with the same parameters
    custom_bn = BatchNorm1dInference(
        num_features=num_features,
        weight=bn.weight.detach(),
        bias=bn.bias.detach(),
        running_mean=bn.running_mean.detach(),
        running_var=bn.running_var.detach(),
        eps=bn.eps,
    )

    # Run both
    with torch.no_grad():
        y_torch = bn(x)
        y_custom = custom_bn.forward(x)

    # Compare
    allclose = torch.allclose(y_torch, y_custom, atol=1e-6, rtol=1e-5)
    max_abs_err = (y_torch - y_custom).abs().max().item()
    max_rel_err = (y_torch - y_custom).abs().max().item() / (y_torch.abs().max().item() + 1e-12)

    print(f"✅ Match: {allclose}")
    print(f"Max absolute error: {max_abs_err:.3e}")
    print(f"Max relative error: {max_rel_err:.3e}")

    # Optional sanity check for first few elements
    print("\nSample output comparison:")
    print("Torch  :", y_torch.flatten()[:5])
    print("Custom :", y_custom.flatten()[:5])

    return allclose, max_abs_err, max_rel_err


# -------------------------------
# Example runs
# -------------------------------
if __name__ == "__main__":
    print("Testing 2D input (N, C):")
    test_batchnorm1d_inference(batch_size=4, num_features=8)

    print("\nTesting 3D input (N, C, L):")
    test_batchnorm1d_inference(batch_size=4, num_features=8, length=10)
"""
