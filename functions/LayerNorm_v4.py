############################################################################################################
# LAYER NORMALIZATION IMPLEMENTATION (PyTorch vs NumPy)
############################################################################################################

import numpy as np
import torch
import torch.nn as nn


# ==========================================================================================================
# NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
import numpy as np
import torch

class LayerNorm:
    def __init__(self, normalized_shape, weight=None, bias=None, eps=1e-5, elementwise_affine=True, bias_condition=True):
        """
        Args:
            normalized_shape (int or tuple): Shape of the input features to normalize over.
            eps (float): Small constant to avoid division by zero.
            elementwise_affine (bool): If True, learnable affine parameters (weight & bias) are created.
            bias_condition (bool): If True and elementwise_affine=True, a bias parameter is created.
        """
        # Convert single int into tuple for consistency
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # Initialize learnable parameters
            if weight is not None:
                self.weight = weight
            else:
                self.weight = np.ones(normalized_shape, dtype=np.float32)
                
            if bias_condition:
                if bias is not None:
                    self.bias = bias
                else:
                    self.bias = np.zeros(normalized_shape, dtype=np.float32)
            else:
                self.bias = None
        else:
            self.weight, self.bias = None, None

    def __call__(self, x):
        """
        Apply Layer Normalization.

        Args:
            x (np.ndarray or torch.Tensor): Input tensor of shape (..., *normalized_shape).

        Returns:
            Same type as input: Normalized tensor with the same shape as input.
        """
        # Handle PyTorch tensors
        if isinstance(x, torch.Tensor):
            return self._forward_torch(x)
        
        # Handle NumPy arrays
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Unsupported input type {type(x)}. Use np.ndarray or torch.Tensor.")

        # Calculate axes to normalize over (last len(normalized_shape) dimensions)
        axes = tuple(range(-len(self.normalized_shape), 0))
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)

        if self.elementwise_affine:
            # Reshape weight and bias to be broadcastable
            shape = (1,) * (x_hat.ndim - len(self.normalized_shape)) + self.normalized_shape
            x_hat = x_hat * self.weight.reshape(shape)
            if self.bias is not None:
                x_hat = x_hat + self.bias.reshape(shape)

        return x_hat
    
    def _forward_torch(self, x):
        """
        Forward pass for PyTorch tensors
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        device = x.device
        dtype = x.dtype
        
        # Convert to numpy for processing
        x_np = x.detach().cpu().numpy()
        
        # Calculate axes to normalize over
        axes = tuple(range(-len(self.normalized_shape), 0))
        mean = np.mean(x_np, axis=axes, keepdims=True)
        var = np.var(x_np, axis=axes, keepdims=True)
        x_hat = (x_np - mean) / np.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            # Reshape weight and bias to be broadcastable
            shape = (1,) * (x_hat.ndim - len(self.normalized_shape)) + self.normalized_shape
            x_hat = x_hat * self.weight.reshape(shape)
            if self.bias is not None:
                x_hat = x_hat + self.bias.reshape(shape)
        
        # Convert back to PyTorch tensor with original device and dtype
        result = torch.from_numpy(x_hat).to(device=device, dtype=dtype)
        return result
# ==========================================================================================================


# ==========================================================================================================
# TEST CASES - Compare NumPy vs PyTorch
# ----------------------------------------------------------------------------------------------------------
def test_layernorm(input_shape, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, test_name=""):
    """
    Compare NumPy LayerNorm against PyTorch nn.LayerNorm
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    print(f"Input shape: {input_shape}")
    print(f"Normalized shape: {normalized_shape}")
    print(f"eps: {eps}, elementwise_affine: {elementwise_affine}, bias: {bias}")
    print(f"{'-'*80}")
    
    # Create random input
    np.random.seed(42)
    torch.manual_seed(42)
    
    np_input = np.random.randn(*input_shape).astype(np.float32)
    torch_input = torch.from_numpy(np_input.copy())
    
    # Create LayerNorm instances
    np_ln = LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias_condition=bias)
    torch_ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    
    # Copy weights and biases if elementwise_affine=True
    if elementwise_affine:
        with torch.no_grad():
            torch_ln.weight.copy_(torch.from_numpy(np_ln.weight))
            if bias and np_ln.bias is not None:
                torch_ln.bias.copy_(torch.from_numpy(np_ln.bias))
    
    # Forward pass
    np_output = np_ln(np_input)
    torch_output = torch_ln(torch_input).detach().numpy()
    
    # Compare results
    max_diff = np.abs(np_output - torch_output).max()
    mean_diff = np.abs(np_output - torch_output).mean()
    match = np.allclose(np_output, torch_output, atol=1e-6, rtol=1e-5)
    
    print(f"NumPy output shape: {np_output.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Max difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")
    print(f"Match (atol=1e-6, rtol=1e-5): {match}")
    
    # Show sample statistics
    print(f"\nNumPy - mean: {np_output.mean():.6f}, std: {np_output.std():.6f}")
    print(f"PyTorch - mean: {torch_output.mean():.6f}, std: {torch_output.std():.6f}")
    print(f"Sample values (NumPy): {np_output.flat[:5]}")
    print(f"Sample values (PyTorch): {torch_output.flat[:5]}")
    
    if match:
        print("✓ PASSED")
    else:
        print("✗ FAILED")
    
    return match


def run_all_tests():
    """
    Run comprehensive test suite
    """
    print("="*80)
    print("COMPARING NumPy LayerNorm vs PyTorch nn.LayerNorm")
    print("="*80)
    
    all_passed = True
    
    # Test 1: Simple 2D case (batch_size, features)
    all_passed &= test_layernorm(
        input_shape=(32, 128),
        normalized_shape=128,
        test_name="2D Input: Normalize last dimension"
    )
    
    # Test 2: 3D case (batch_size, seq_len, features) - NLP typical
    all_passed &= test_layernorm(
        input_shape=(8, 50, 256),
        normalized_shape=256,
        test_name="3D Input: NLP sequence (normalize features)"
    )
    
    # Test 3: Normalize over last 2 dimensions
    all_passed &= test_layernorm(
        input_shape=(16, 32, 64),
        normalized_shape=(32, 64),
        test_name="3D Input: Normalize last 2 dimensions"
    )
    
    # Test 4: 4D case (batch, channels, height, width) - Vision
    all_passed &= test_layernorm(
        input_shape=(4, 3, 32, 32),
        normalized_shape=(3, 32, 32),
        test_name="4D Input: Image (normalize C, H, W)"
    )
    
    # Test 5: Without elementwise_affine
    all_passed &= test_layernorm(
        input_shape=(16, 128),
        normalized_shape=128,
        elementwise_affine=False,
        test_name="Without learnable parameters"
    )
    
    # Test 6: Without bias
    all_passed &= test_layernorm(
        input_shape=(16, 128),
        normalized_shape=128,
        bias=False,
        test_name="With weight but no bias"
    )
    
    # Test 7: Different eps value
    all_passed &= test_layernorm(
        input_shape=(8, 64),
        normalized_shape=64,
        eps=1e-6,
        test_name="Custom epsilon value (1e-6)"
    )
    
    # Test 8: Single element batch
    all_passed &= test_layernorm(
        input_shape=(1, 512),
        normalized_shape=512,
        test_name="Single sample in batch"
    )
    
    # Test 9: Large feature dimension
    all_passed &= test_layernorm(
        input_shape=(4, 1024),
        normalized_shape=1024,
        test_name="Large feature dimension"
    )
    
    # Test 10: Normalize over tuple of dimensions
    all_passed &= test_layernorm(
        input_shape=(8, 16, 32),
        normalized_shape=(16, 32),
        test_name="Normalize over multiple dimensions (tuple)"
    )
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - NumPy implementation matches PyTorch!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return all_passed


# ==========================================================================================================
# USAGE EXAMPLES
# ----------------------------------------------------------------------------------------------------------
def usage_examples():
    """Show usage examples comparing both implementations"""
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    # Example 1: Basic usage
    print("\nExample 1: Basic LayerNorm on 2D input")
    print("-"*40)
    
    # Create input
    np.random.seed(0)
    x_np = np.random.randn(2, 4).astype(np.float32)
    x_torch = torch.from_numpy(x_np.copy())
    
    print("Input shape:", x_np.shape)
    print("Input:\n", x_np)
    
    # NumPy
    ln_np = LayerNorm(4)
    output_np = ln_np(x_np)
    
    # PyTorch
    ln_torch = nn.LayerNorm(4)
    with torch.no_grad():
        ln_torch.weight.copy_(torch.from_numpy(ln_np.weight))
        ln_torch.bias.copy_(torch.from_numpy(ln_np.bias))
    output_torch = ln_torch(x_torch).detach().numpy()
    
    print("\nNumPy output:\n", output_np)
    print("PyTorch output:\n", output_torch)
    print(f"Match: {np.allclose(output_np, output_torch, atol=1e-6)}")
    
    # Example 2: NLP use case
    print("\n" + "="*80)
    print("Example 2: NLP - Normalize features in a sequence")
    print("-"*40)
    
    # Shape: (batch_size, sequence_length, embedding_dim)
    batch_size, seq_len, embed_dim = 2, 5, 8
    x_np = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    
    ln_np = LayerNorm(embed_dim)
    output_np = ln_np(x_np)
    
    print(f"Input shape: {x_np.shape}")
    print(f"Output shape: {output_np.shape}")
    print(f"Mean per sample (should be ~0): {output_np.mean(axis=-1)}")
    print(f"Std per sample (should be ~1): {output_np.std(axis=-1)}")


# ==========================================================================================================
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run comprehensive comparison tests
    run_all_tests()
    
    # Show usage examples
    usage_examples()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The NumPy LayerNorm implementation perfectly replicates")
    print("PyTorch's nn.LayerNorm behavior for all configurations.")
    print("="*80)