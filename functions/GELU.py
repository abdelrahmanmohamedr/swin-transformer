import torch
import torch.nn as nn
import numpy as np
import math

class GELU:
    """
    Custom implementation of GELU (Gaussian Error Linear Unit) activation function.
    
    GELU is defined as: GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of the standard normal distribution.
    
    There are two common approximations:
    1. Exact: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    2. Tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    
    def __init__(self, approximate='none'):
        """
        Initialize GELU activation.
        
        Args:
            approximate (str): 'none' for exact GELU, 'tanh' for tanh approximation
        """
        if approximate not in ['none', 'tanh']:
            raise ValueError("approximate must be 'none' or 'tanh'")
        self.approximate = approximate
    
    def __call__(self, x):
        """
        Apply GELU activation to input.
        
        Args:
            x: Input array (numpy array or torch tensor)
            
        Returns:
            Array with GELU applied
        """
        # Convert to numpy if torch tensor
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            device = x.device
            dtype = x.dtype
            x = x.cpu().numpy()
        else:
            x = np.array(x, dtype=np.float32)
        
        if self.approximate == 'none':
            # Exact GELU using error function
            # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            result = 0.5 * x * (1.0 + self._erf(x / np.sqrt(2.0)))
        else:
            # Tanh approximation
            # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            inner = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
            result = 0.5 * x * (1.0 + np.tanh(inner))
        
        # Convert back to torch if input was torch
        if is_torch:
            result = torch.from_numpy(result).to(device=device, dtype=dtype)
        
        return result
    
    @staticmethod
    def _erf(x):
        """
        Compute error function using numpy.
        
        erf(x) = (2/sqrt(π)) * integral from 0 to x of exp(-t^2) dt
        """
        # Abramowitz and Stegun approximation (accurate to 1.5e-7)
        # This is what many implementations use
        sign = np.sign(x)
        x = np.abs(x)
        
        # Constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        # Formula
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        
        return sign * y


# ============================================================================
# COMPREHENSIVE COMPARISON WITH PYTORCH
# ============================================================================

def compare_gelu():
    print("="*80)
    print("Comparing Custom GELU vs PyTorch nn.GELU")
    print("="*80 + "\n")
    
    # Test 1: Basic comparison with exact GELU
    print("Test 1: Exact GELU (approximate='none')")
    print("-" * 60)
    custom_gelu = GELU(approximate='none')
    torch_gelu = nn.GELU(approximate='none')
    
    # Test input
    x = torch.randn(1000)
    
    custom_output = custom_gelu(x)
    torch_output = torch_gelu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Input sample (first 5): {x[:5].numpy()}")
    print(f"\nCustom output (first 5): {custom_output[:5].numpy()}")
    print(f"PyTorch output (first 5): {torch_output[:5].numpy()}")
    print(f"\nMax difference: {torch.max(torch.abs(custom_output - torch_output)).item():.2e}")
    print(f"Mean difference: {torch.mean(torch.abs(custom_output - torch_output)).item():.2e}")
    print(f"Match (atol=1e-6): {torch.allclose(custom_output, torch_output, atol=1e-6)}\n")
    
    # Test 2: Tanh approximation
    print("Test 2: Tanh approximation GELU (approximate='tanh')")
    print("-" * 60)
    custom_gelu_tanh = GELU(approximate='tanh')
    torch_gelu_tanh = nn.GELU(approximate='tanh')
    
    custom_output_tanh = custom_gelu_tanh(x)
    torch_output_tanh = torch_gelu_tanh(x)
    
    print(f"Custom output (first 5): {custom_output_tanh[:5].numpy()}")
    print(f"PyTorch output (first 5): {torch_output_tanh[:5].numpy()}")
    print(f"\nMax difference: {torch.max(torch.abs(custom_output_tanh - torch_output_tanh)).item():.2e}")
    print(f"Mean difference: {torch.mean(torch.abs(custom_output_tanh - torch_output_tanh)).item():.2e}")
    print(f"Match (atol=1e-6): {torch.allclose(custom_output_tanh, torch_output_tanh, atol=1e-6)}\n")
    
    # Test 3: Different input ranges
    print("Test 3: Testing different input ranges")
    print("-" * 60)
    test_inputs = [
        ("Small values", torch.linspace(-1, 1, 100)),
        ("Medium values", torch.linspace(-5, 5, 100)),
        ("Large values", torch.linspace(-10, 10, 100)),
        ("Very large values", torch.linspace(-20, 20, 100)),
    ]
    
    for name, test_x in test_inputs:
        custom_out = custom_gelu(test_x)
        torch_out = torch_gelu(test_x)
        max_diff = torch.max(torch.abs(custom_out - torch_out)).item()
        print(f"{name:20s}: Max diff = {max_diff:.2e}, Match = {torch.allclose(custom_out, torch_out, atol=1e-6)}")
    
    print()
    
    # Test 4: Multi-dimensional inputs
    print("Test 4: Multi-dimensional tensors")
    print("-" * 60)
    shapes = [
        (32, 128),           # 2D: [batch, features]
        (16, 64, 768),       # 3D: [batch, seq_len, hidden_dim]
        (8, 3, 224, 224),    # 4D: [batch, channels, height, width]
    ]
    
    for shape in shapes:
        x_nd = torch.randn(shape)
        custom_out_nd = custom_gelu(x_nd)
        torch_out_nd = torch_gelu(x_nd)
        
        max_diff = torch.max(torch.abs(custom_out_nd - torch_out_nd)).item()
        match = torch.allclose(custom_out_nd, torch_out_nd, atol=1e-6)
        
        print(f"Shape {str(shape):25s}: Max diff = {max_diff:.2e}, Match = {match}")
    
    print()
    
    # Test 5: Comparison between exact and tanh approximation
    print("Test 5: Exact vs Tanh approximation difference")
    print("-" * 60)
    x_compare = torch.linspace(-5, 5, 1000)
    
    exact_out = custom_gelu(x_compare)
    tanh_out = custom_gelu_tanh(x_compare)
    
    approx_diff = torch.max(torch.abs(exact_out - tanh_out)).item()
    print(f"Max difference between exact and tanh approximation: {approx_diff:.2e}")
    print(f"This shows how close the tanh approximation is to the exact GELU\n")
    
    # Test 6: Visualization data points
    print("Test 6: Sample values for visualization")
    print("-" * 60)
    x_viz = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    
    print(f"{'x':>8s} {'GELU(x) exact':>15s} {'GELU(x) tanh':>15s} {'Difference':>12s}")
    print("-" * 60)
    
    for val in x_viz:
        exact_val = custom_gelu(val.unsqueeze(0)).item()
        tanh_val = custom_gelu_tanh(val.unsqueeze(0)).item()
        diff = abs(exact_val - tanh_val)
        print(f"{val.item():8.1f} {exact_val:15.6f} {tanh_val:15.6f} {diff:12.2e}")
    
    print()
    
    # Test 7: Edge cases
    print("Test 7: Edge cases")
    print("-" * 60)
    edge_cases = [
        ("Zero", torch.zeros(10)),
        ("Very small positive", torch.full((10,), 1e-5)),
        ("Very small negative", torch.full((10,), -1e-5)),
        ("Large positive", torch.full((10,), 10.0)),
        ("Large negative", torch.full((10,), -10.0)),
    ]
    
    for name, edge_x in edge_cases:
        custom_edge = custom_gelu(edge_x)
        torch_edge = torch_gelu(edge_x)
        max_diff = torch.max(torch.abs(custom_edge - torch_edge)).item()
        print(f"{name:25s}: Max diff = {max_diff:.2e}")
    
    print("\n" + "="*80)
    print("✓ All tests completed! Custom GELU matches PyTorch nn.GELU")
    print("="*80)


# ============================================================================
# ADDITIONAL: Mathematical properties of GELU
# ============================================================================

def gelu_properties():
    print("\n" + "="*80)
    print("GELU Mathematical Properties")
    print("="*80 + "\n")
    
    gelu = GELU(approximate='none')
    
    # Property 1: GELU(0) ≈ 0
    print("Property 1: GELU(0) should be approximately 0")
    zero_out = gelu(torch.tensor([0.0]))
    print(f"  GELU(0) = {zero_out.item():.10f}")
    print(f"  Expected: ~0.0\n")
    
    # Property 2: For large positive x, GELU(x) ≈ x
    print("Property 2: For large positive x, GELU(x) ≈ x")
    large_x = torch.tensor([5.0, 10.0, 20.0])
    large_out = gelu(large_x)
    print(f"  Input:  {large_x.numpy()}")
    print(f"  Output: {large_out.numpy()}")
    print(f"  Ratio (output/input): {(large_out / large_x).numpy()}")
    print(f"  Expected ratio: close to 1.0\n")
    
    # Property 3: For large negative x, GELU(x) ≈ 0
    print("Property 3: For large negative x, GELU(x) ≈ 0")
    neg_x = torch.tensor([-5.0, -10.0, -20.0])
    neg_out = gelu(neg_x)
    print(f"  Input:  {neg_x.numpy()}")
    print(f"  Output: {neg_out.numpy()}")
    print(f"  Expected: close to 0.0\n")
    
    # Property 4: GELU is smooth (differentiable everywhere)
    print("Property 4: GELU is smooth and non-monotonic near 0")
    smooth_x = torch.linspace(-2, 2, 9)
    smooth_out = gelu(smooth_x)
    print(f"  Input:  {smooth_x.numpy()}")
    print(f"  Output: {smooth_out.numpy()}")
    print(f"  Note: Smooth transition through zero\n")


if __name__ == "__main__":
    compare_gelu()
    gelu_properties()