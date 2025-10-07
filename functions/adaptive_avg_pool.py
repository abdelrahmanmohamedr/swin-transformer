import math
import torch
import torch.nn as nn
import numpy as np

class AdaptiveAvgPool1d:
    def __init__(self, output_size):
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.output_size = output_size

    def _pool_1d(self, data):
        """Pool a single 1D sequence"""
        input_size = len(data)
        result = []
        
        for i in range(self.output_size):
            # Calculate start and end indices for this output position
            start = int(math.floor(i * input_size / self.output_size))
            end = int(math.ceil((i + 1) * input_size / self.output_size))
            
            # Extract segment and compute average
            segment = data[start:end]
            avg = sum(segment) / len(segment) if segment else 0.0
            result.append(avg)
        
        return result

    def __call__(self, data):
        """
        Apply adaptive average pooling to input data
        Supports both PyTorch tensors and Python lists:
        - PyTorch Tensors: [N, C, L] or [C, L]
        - Python Lists: [L], [C, L], or [N, C, L]
        """
        # Handle PyTorch tensors
        if isinstance(data, torch.Tensor):
            return self._forward_torch(data)
        
        # Handle Python lists (original implementation)
        # Handle empty input
        if not data:
            raise ValueError("Input data cannot be empty")
        
        # 1D input: [L]
        if isinstance(data[0], (int, float)):
            return self._pool_1d(data)

        # 2D input: [C, L]
        elif isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], (int, float)):
            return [self._pool_1d(channel) for channel in data]

        # 3D input: [N, C, L]
        elif (isinstance(data[0], list) and len(data[0]) > 0 and 
              isinstance(data[0][0], list) and len(data[0][0]) > 0 and 
              isinstance(data[0][0][0], (int, float))):
            return [
                [self._pool_1d(channel) for channel in sample]
                for sample in data
            ]

        else:
            raise ValueError("Unsupported input format. Expected 1D, 2D, or 3D nested lists of numbers or PyTorch tensor")
    
    def _forward_torch(self, x):
        """
        Forward pass for PyTorch tensors using built-in adaptive_avg_pool1d
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, L] or [C, L]
        
        Returns:
            torch.Tensor: Pooled tensor
        """
        return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size)


def test_adaptive_pool(input_data, output_size, test_name):
    """
    Compare custom AdaptiveAvgPool1d with PyTorch's nn.AdaptiveAvgPool1d
    """
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    # Custom implementation
    custom_pool = AdaptiveAvgPool1d(output_size)
    custom_result = custom_pool(input_data)
    
    # PyTorch implementation
    torch_pool = nn.AdaptiveAvgPool1d(output_size)
    
    # Convert input to torch tensor
    if isinstance(input_data[0], (int, float)):
        # 1D case: need to add batch and channel dimensions
        torch_input = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        torch_result = torch_pool(torch_input).squeeze().tolist()
    elif isinstance(input_data[0], list) and isinstance(input_data[0][0], (int, float)):
        # 2D case: [C, L] -> add batch dimension
        torch_input = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # [1, C, L]
        torch_result = torch_pool(torch_input).squeeze(0).tolist()
    else:
        # 3D case: [N, C, L]
        torch_input = torch.tensor(input_data, dtype=torch.float32)
        torch_result = torch_pool(torch_input).tolist()
    
    # Print shapes
    print(f"Input shape: {np.array(input_data).shape}")
    print(f"Output size: {output_size}")
    print(f"Custom output shape: {np.array(custom_result).shape}")
    print(f"Torch output shape: {np.array(torch_result).shape}")
    
    # Compare results
    custom_flat = np.array(custom_result).flatten()
    torch_flat = np.array(torch_result).flatten()
    
    max_diff = np.abs(custom_flat - torch_flat).max()
    print(f"\nMax difference: {max_diff:.10f}")
    
    # Check if results match
    is_close = np.allclose(custom_flat, torch_flat, atol=1e-6)
    print(f"Results match: {is_close}")
    
    if not is_close:
        print("\nCustom result:")
        print(custom_result)
        print("\nTorch result:")
        print(torch_result)
    
    return is_close


def test_tensor_input():
    """Test that the custom pool works with PyTorch tensors"""
    print("\n" + "="*60)
    print("Testing PyTorch Tensor Input")
    print("="*60)
    
    custom_pool = AdaptiveAvgPool1d(3)
    torch_pool = nn.AdaptiveAvgPool1d(3)
    
    # Test with tensor input
    tensor_input = torch.randn(2, 4, 10)  # [N, C, L]
    print(f"Input tensor shape: {tensor_input.shape}")
    
    # Custom
    custom_output = custom_pool(tensor_input)
    print(f"Custom output shape: {custom_output.shape}")
    
    # PyTorch
    torch_output = torch_pool(tensor_input)
    print(f"PyTorch output shape: {torch_output.shape}")
    
    # Compare
    max_diff = torch.max(torch.abs(custom_output - torch_output)).item()
    print(f"Max difference: {max_diff:.10f}")
    
    is_close = torch.allclose(custom_output, torch_output, atol=1e-6)
    print(f"Results match: {is_close}")
    
    return is_close


if __name__ == "__main__":
    print("Testing AdaptiveAvgPool1d Implementation")
    print("="*60)
    
    all_passed = True
    
    # Original list-based tests
    data_1d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    all_passed &= test_adaptive_pool(data_1d, 3, "1D: 6 -> 3")
    
    data_2d = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    ]
    all_passed &= test_adaptive_pool(data_2d, 3, "2D: [3, 6] -> [3, 3]")
    
    data_3d = [
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        ],
        [
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]
    ]
    all_passed &= test_adaptive_pool(data_3d, 2, "3D: [2, 2, 4] -> [2, 2, 2]")
    
    # New tensor-based test
    all_passed &= test_tensor_input()
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*60)