import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple

class ManualConv2d(nn.Module):
    """
    Fully Manual Conv2d with explicit img2col and matmul operations.
    
    - Manual img2col with explicit loops
    - Manual matmul with explicit control
    - Full control over every operation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        super(ManualConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def img2col(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Convert input tensor to column matrix for convolution.
        
        Args:
            input_tensor: [batch, in_channels, height, width]
        
        Returns:
            col_matrix: [batch, num_patches, patch_size]
            out_height: int - Output height
            out_width: int - Output width
        """
        batch_size, in_channels, height, width = input_tensor.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation
        
        # Calculate output dimensions
        out_height = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(
                input_tensor, (pad_w, pad_w, pad_h, pad_h), 
                mode=self.padding_mode, value=0
            )
        
        # Create 3D output matrix preserving batch dimension
        num_patches = out_height * out_width
        patch_size = in_channels * kernel_h * kernel_w
        
        col_matrix = torch.zeros(
            batch_size,    # Keep batch dimension separate!
            num_patches,   # Total patches per image
            patch_size,    # Flattened patch size
            device=input_tensor.device,
            dtype=input_tensor.dtype
        )
        
        # Extract patches for each batch separately  
        for b in range(batch_size):
            patch_idx = 0
            for out_y in range(out_height):
                for out_x in range(out_width):
                    # Calculate input region for this output position
                    y_start = out_y * stride_h
                    x_start = out_x * stride_w
                    
                    # Extract kernel-sized patch and flatten it
                    flat_idx = 0
                    for c in range(in_channels):
                        for ky in range(kernel_h):
                            for kx in range(kernel_w):
                                y_pos = y_start + ky * dil_h
                                x_pos = x_start + kx * dil_w
                                
                                # Extract value (with bounds checking for padded input)
                                if (0 <= y_pos < input_tensor.shape[2] and 
                                    0 <= x_pos < input_tensor.shape[3]):
                                    col_matrix[b, patch_idx, flat_idx] = input_tensor[b, c, y_pos, x_pos]
                                # else: remains 0 (padding)
                                
                                flat_idx += 1
                    
                    patch_idx += 1
        
        return col_matrix, out_height, out_width
    
    def matmul_manual(self, col_matrix: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Manual matrix multiplication with explicit control.
        
        Args:
            col_matrix: [batch, num_patches, patch_size]
            weight_matrix: [out_channels, patch_size]
        Returns:
            result: [batch, num_patches, out_channels]
        """
        batch_size, num_patches, patch_size = col_matrix.shape
        out_channels = weight_matrix.shape[0]
        
        result = torch.zeros(batch_size, num_patches, out_channels,
                           device=col_matrix.device, dtype=col_matrix.dtype)
        
        # MANUAL MATRIX MULTIPLICATION with controlled loops
        for b in range(batch_size):
            batch_patches = col_matrix[b]  # [num_patches, patch_size]
            
            for out_ch in range(out_channels):
                filter_weights = weight_matrix[out_ch]  # [patch_size]
                
                # OPTION 1: Vectorized dot product (faster)
                result[b, :, out_ch] = torch.sum(batch_patches * filter_weights.unsqueeze(0), dim=1)
                
                # OPTION 2: Explicit inner loops (full control, slower)
                # Uncomment for complete manual control:
                # for patch_idx in range(num_patches):
                #     accumulator = 0.0
                #     for k in range(patch_size):
                #         accumulator += batch_patches[patch_idx, k] * filter_weights[k]
                #     result[b, patch_idx, out_ch] = accumulator
        
        return result
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with manual operations.
        """
        # Step 1: Manual img2col with explicit loops
        col_matrix, out_height, out_width = self.img2col(input)
        
        # Step 2: Reshape weights
        weight_matrix = self.weight.view(self.out_channels, -1)
        
        # Step 3: Manual matrix multiplication  
        result = self.matmul_manual(col_matrix, weight_matrix)
        
        # Step 4: Add bias
        if self.bias is not None:
            result = result + self.bias.view(1, 1, -1)
        
        # Step 5: Reshape to conv output format
        output = result.view(input.shape[0], out_height, out_width, self.out_channels)
        output = output.permute(0, 3, 1, 2)
        
        return output

