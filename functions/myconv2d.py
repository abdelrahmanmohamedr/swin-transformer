import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True
    ):
        super(MyConv2d, self).__init__()

        # Store params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.has_bias = bias

        # Define weights and bias as nn.Parameter (so optimizers see them)
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape) * np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size))))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        w_np = self.weight.detach().cpu().numpy()
        b_np = self.bias.detach().cpu().numpy() if self.bias is not None else None

        N, C_in, H, W = x_np.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        # Output shape
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        # Pad input
        x_padded = np.pad(
            x_np,
            ((0, 0), (0, 0), (pH, pH), (pW, pW)),
            mode="constant"
        )

        # Output array
        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)

        # Grouped convolution
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups

        for n in range(N):
            for g in range(self.groups):
                in_start = g * in_per_group
                in_end = (g + 1) * in_per_group
                out_start = g * out_per_group
                out_end = (g + 1) * out_per_group

                for oc in range(out_per_group):
                    oc_global = out_start + oc
                    for ic in range(in_per_group):
                        ic_global = in_start + ic
                        for i in range(H_out):
                            for j in range(W_out):
                                h_start = i * sH
                                w_start = j * sW
                                h_end = h_start + dH * (kH - 1) + 1
                                w_end = w_start + dW * (kW - 1) + 1

                                patch = x_padded[n, ic_global, h_start:h_end:dH, w_start:w_end:dW]
                                out[n, oc_global, i, j] += np.sum(patch * w_np[oc_global, ic])

                    if b_np is not None:
                        out[n, oc_global, :, :] += b_np[oc_global]

        return torch.from_numpy(out).to(x.device)

import torch
import torch.nn as nn
import numpy as np

# Import your class
# from my_conv import MyConv2d   # if you put it in a file
from torch import manual_seed
manual_seed(0)

# Define test parameters
in_channels = 3
out_channels = 4
kernel_size = 3
stride = 2
padding = 1
dilation = 1
groups = 1
bias = True

# Create input
x = torch.randn(10, in_channels, 224, 224)  # (N, C, H, W)

# Create reference Conv2d
conv_ref = nn.Conv2d(
    in_channels, out_channels, kernel_size,
    stride=stride, padding=padding,
    dilation=dilation, groups=groups, bias=bias
)

# Create MyConv2d
conv_my = MyConv2d(
    in_channels, out_channels, kernel_size,
    stride=stride, padding=padding,
    dilation=dilation, groups=groups, bias=bias
)

# Copy weights & biases
with torch.no_grad():
    conv_my.weight.copy_(conv_ref.weight)
    if bias:
        conv_my.bias.copy_(conv_ref.bias)

# Run both
out_ref = conv_ref(x)
out_my = conv_my(x)

# Compare
print("Reference output shape:", out_ref.shape)
print("MyConv2d output shape:", out_my.shape)

# Maximum absolute difference
diff = torch.abs(out_ref - out_my).max()
print("Max difference:", diff.item())

import torch
import torch.nn as nn

def test_case(N, C_in, C_out, H, W, k, stride=1, pad=0, dil=1, groups=1, bias=True):
    print(f"\n=== Test: N={N}, Cin={C_in}, Cout={C_out}, HxW={H}x{W}, k={k}, stride={stride}, pad={pad}, dil={dil}, groups={groups}, bias={bias} ===")
    x = torch.randn(N, C_in, H, W)

    conv_ref = nn.Conv2d(
        C_in, C_out, k,
        stride=stride, padding=pad,
        dilation=dil, groups=groups, bias=bias
    )
    conv_my = MyConv2d(
        C_in, C_out, k,
        stride=stride, padding=pad,
        dilation=dil, groups=groups, bias=bias
    )

    with torch.no_grad():
        conv_my.weight.copy_(conv_ref.weight)
        if bias:
            conv_my.bias.copy_(conv_ref.bias)

    out_ref = conv_ref(x)
    out_my = conv_my(x)

    print("  Ref shape:", tuple(out_ref.shape))
    print("  My shape :", tuple(out_my.shape))
    diff = torch.abs(out_ref - out_my).max().item()
    print("  Max diff :", diff)
    print("  Match?   :", torch.allclose(out_ref, out_my, atol=1e-5))


if __name__ == "__main__":
    # --- Simple sanity checks ---
    test_case(1, 1, 1, 5, 5, 3)                 # single-channel, single filter
    test_case(2, 3, 4, 8, 8, 3)                 # small image, multi-in/out

    # --- Stride and padding ---
    test_case(1, 3, 2, 7, 7, 3, stride=2)       # stride > 1
    test_case(1, 3, 2, 7, 7, 3, pad=1)          # padding
    test_case(1, 3, 2, 7, 7, 3, stride=2, pad=1)

    # --- Dilation ---
    test_case(1, 3, 2, 10, 10, 3, dil=2)        # dilation

    # --- Groups ---
    test_case(1, 4, 4, 6, 6, 3, groups=2)       # grouped conv
    test_case(1, 4, 4, 6, 6, 3, groups=4)       # depthwise conv

    # --- Bias/no-bias ---
    test_case(1, 3, 3, 6, 6, 3, bias=False)

    # --- Tiny images ---
    test_case(1, 1, 1, 1, 1, 1)                 # 1x1 input, 1x1 kernel
    test_case(1, 1, 1, 2, 2, 3, pad=1)          # padding makes conv possible