import numpy as np
# Import manually implemented primitives
from GELU import GELU
from linear import ExplicitLinear
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from original_swin import Mlp

class ManualMlp:
    """
    Manual, NumPy-based implementation of the Swin Transformer MLP block (Feed-Forward Network).
    This block performs the expansion, activation, and contraction steps.
    """
    def __init__(self, W1, B1, W2, B2, in_features=None, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        # Extract dimensions from weight shapes
        in_features = W1.shape[1]
        hidden_features = W1.shape[0]
        out_features = W2.shape[0]
        
        # FC1: expansion layer with provided weights
        self.fc1 = ExplicitLinear(
            in_features=in_features,
            out_features=hidden_features,
            weight=W1,
            bias=B1
        )
        
        # Activation
        self.act = act_layer()
        
        # FC2: contraction layer with provided weights
        self.fc2 = ExplicitLinear(
            in_features=hidden_features,
            out_features=out_features,
            weight=W2,
            bias=B2
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    def __call__(self, x):
        """Make the class callable"""
        return self.forward(x)

    def window_partition(x, window_size):
        """Partition into non-overlapping windows - Pure NumPy"""
        # Convert to numpy if it's a tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        B, H, W, C = x.shape
        x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
        # NumPy transpose accepts tuple of axes
        windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
        return windows

    def window_reverse(windows, window_size, H, W):
        """Reverse window partition - Pure NumPy"""
        # Convert to numpy if it's a tensor
        if isinstance(windows, torch.Tensor):
            windows = windows.detach().cpu().numpy()

        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
        # NumPy transpose accepts tuple of axes
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
        return x
    
class ManualMlpTestSuite:
    """
    Comprehensive test suite to compare PyTorch Mlp with Manual NumPy ManualMlp
    """
    
    @staticmethod
    def extract_pytorch_weights(mlp_module: Mlp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract weights and biases from PyTorch Mlp module
        
        Returns:
            W1, B1, W2, B2 as numpy arrays with correct shapes for ManualMlp
        """
        # Extract fc1 weights and biases
        W1 = mlp_module.fc1.weight.detach().cpu().numpy()  # [hidden_features, in_features]
        B1 = mlp_module.fc1.bias.detach().cpu().numpy()    # [hidden_features]
        
        # Extract fc2 weights and biases
        W2 = mlp_module.fc2.weight.detach().cpu().numpy()  # [out_features, hidden_features]
        B2 = mlp_module.fc2.bias.detach().cpu().numpy()    # [out_features]
        
        return W1, B1, W2, B2
    
    @staticmethod
    def create_test_input(batch_size: int, seq_len: int, in_features: int, seed: int = 42) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Create identical test input for both PyTorch and NumPy implementations
        
        Args:
            batch_size: Batch size (B)
            seq_len: Sequence length (L) - number of tokens
            in_features: Input feature dimension (C)
            seed: Random seed for reproducibility
            
        Returns:
            PyTorch tensor and NumPy array with identical values
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create random input
        x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
        x_torch = torch.from_numpy(x_np.copy())
        
        return x_torch, x_np
    
    @staticmethod
    def compare_outputs(output_pytorch: np.ndarray, output_manual: np.ndarray, 
                       tolerance: float = 1e-5, test_name: str = "Test") -> dict:
        """
        Compare outputs from PyTorch and Manual implementations
        
        Returns:
            Dictionary with comparison results
        """
        # Compute differences
        abs_diff = np.abs(output_pytorch - output_manual)
        rel_diff = abs_diff / (np.abs(output_pytorch) + 1e-8)
        
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        passed = max_abs_diff < tolerance
        
        results = {
            'test_name': test_name,
            'passed': passed,
            'max_abs_diff': max_abs_diff,
            'mean_abs_diff': mean_abs_diff,
            'max_rel_diff': max_rel_diff,
            'mean_rel_diff': mean_rel_diff,
            'tolerance': tolerance,
            'output_shape': output_pytorch.shape
        }
        
        return results
    
    @staticmethod
    def print_results(results: dict):
        """Print test results in a formatted way"""
        status = "✓ PASSED" if results['passed'] else "✗ FAILED"
        print(f"\n{'-'*70}")
        print(f"Test: {results['test_name']} - {status}")
        print(f"{'-'*70}")
        print(f"Output Shape:        {results['output_shape']}")
        print(f"Max Absolute Diff:   {results['max_abs_diff']:.2e} (tolerance: {results['tolerance']:.2e})")
        print(f"Mean Absolute Diff:  {results['mean_abs_diff']:.2e}")
        print(f"Max Relative Diff:   {results['max_rel_diff']:.2e}")
        print(f"Mean Relative Diff:  {results['mean_rel_diff']:.2e}")
        print(f"{'-'*70}\n")


def run_all_tests(ManualMlp):
    """
    Run comprehensive test suite comparing PyTorch Mlp and Manual ManualMlp
    
    Args:
        ManualMlp: Your manual MLP implementation class
    """
    print("="*70)
    print("MLP COMPARISON TEST SUITE")
    print("Comparing Original Swin Transformer Mlp vs Manual NumPy Implementation")
    print("="*70)
    
    test_suite = ManualMlpTestSuite()
    all_results = []
    
    # Test configurations (batch_size, seq_len, in_features, hidden_features, test_name)
    test_configs = [
        (1, 49, 96, 384, "Test 1: Single Batch, 7x7 patches, dim=96"),
        (2, 196, 192, 768, "Test 2: Small Batch, 14x14 patches, dim=192"),
        (4, 784, 384, 1536, "Test 3: Medium Batch, 28x28 patches, dim=384"),
        (1, 3136, 768, 3072, "Test 4: Large Resolution, 56x56 patches, dim=768"),
        (8, 49, 96, 384, "Test 5: Large Batch, 7x7 patches, dim=96"),
    ]
    
    for batch_size, seq_len, in_features, hidden_features, test_name in test_configs:
        print(f"\nRunning: {test_name}")
        print(f"Config: B={batch_size}, L={seq_len}, C={in_features}, Hidden={hidden_features}")
        
        # Create PyTorch Mlp using the original Swin Transformer implementation
        mlp_pytorch = Mlp(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=in_features,
            act_layer=nn.GELU,
            drop=0.0  # No dropout for testing
        )
        mlp_pytorch.eval()  # Set to evaluation mode
        
        # Extract weights from PyTorch Mlp
        W1, B1, W2, B2 = test_suite.extract_pytorch_weights(mlp_pytorch)
        
        # Create Manual Mlp with the same weights
        mlp_manual = ManualMlp(W1, B1, W2, B2)
        
        # Create identical test input
        x_torch, x_np = test_suite.create_test_input(batch_size, seq_len, in_features)
        
        # PyTorch forward pass (no gradient computation)
        with torch.no_grad():
            output_pytorch = mlp_pytorch(x_torch).cpu().numpy()
        
        # Manual forward pass (reshape to 2D if needed)
        x_np_2d = x_np.reshape(-1, in_features)
        output_manual = mlp_manual(x_np_2d)
        output_manual = output_manual.reshape(batch_size, seq_len, in_features)
        
        # Compare outputs
        results = test_suite.compare_outputs(
            output_pytorch, output_manual, 
            tolerance=1e-4, test_name=test_name
        )
        
        test_suite.print_results(results)
        all_results.append(results)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for r in all_results if r['passed'])
    total = len(all_results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed successfully!")
        print("Your ManualMlp implementation matches the PyTorch version!")
    else:
        print("✗ Some tests failed. Please review the results above.")
        print("Check your ExplicitLinear and GELU implementations.")
    
    print("="*70 + "\n")
    
    return all_results


def test_edge_cases(ManualMlp):
    """
    Test edge cases and special scenarios
    """
    print("\n" + "="*70)
    print("EDGE CASE TESTS")
    print("="*70)
    
    test_suite = ManualMlpTestSuite()
    
    # Edge Case 1: Very small values (near zero)
    print("\n1. Testing with near-zero input values...")
    in_features, hidden_features = 96, 384
    
    mlp_pytorch = Mlp(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=in_features,
        drop=0.0
    )
    mlp_pytorch.eval()
    
    W1, B1, W2, B2 = test_suite.extract_pytorch_weights(mlp_pytorch)
    mlp_manual = ManualMlp(W1, B1, W2, B2)
    
    x_torch = torch.randn(1, 49, in_features) * 1e-6
    x_np = x_torch.numpy()
    
    with torch.no_grad():
        out_torch = mlp_pytorch(x_torch).numpy()
    out_manual = mlp_manual(x_np.reshape(-1, in_features)).reshape(1, 49, in_features)
    
    results = test_suite.compare_outputs(out_torch, out_manual, tolerance=1e-5, 
                                        test_name="Near-zero inputs")
    test_suite.print_results(results)
    
    # Edge Case 2: Large magnitude values
    print("2. Testing with large magnitude input values...")
    x_torch = torch.randn(1, 49, in_features) * 100
    x_np = x_torch.numpy()
    
    with torch.no_grad():
        out_torch = mlp_pytorch(x_torch).numpy()
    out_manual = mlp_manual(x_np.reshape(-1, in_features)).reshape(1, 49, in_features)
    
    results = test_suite.compare_outputs(out_torch, out_manual, tolerance=1e-3,
                                        test_name="Large magnitude inputs")
    test_suite.print_results(results)
    
    # Edge Case 3: All zeros
    print("3. Testing with all-zero input...")
    x_torch = torch.zeros(1, 49, in_features)
    x_np = x_torch.numpy()
    
    with torch.no_grad():
        out_torch = mlp_pytorch(x_torch).numpy()
    out_manual = mlp_manual(x_np.reshape(-1, in_features)).reshape(1, 49, in_features)
    
    results = test_suite.compare_outputs(out_torch, out_manual, tolerance=1e-6,
                                        test_name="All-zero inputs")
    test_suite.print_results(results)
    
    # Edge Case 4: Single token
    print("4. Testing with single token...")
    x_torch = torch.randn(1, 1, in_features)
    x_np = x_torch.numpy()
    
    with torch.no_grad():
        out_torch = mlp_pytorch(x_torch).numpy()
    out_manual = mlp_manual(x_np.reshape(-1, in_features)).reshape(1, 1, in_features)
    
    results = test_suite.compare_outputs(out_torch, out_manual, tolerance=1e-5,
                                        test_name="Single token")
    test_suite.print_results(results)
    
    print("="*70 + "\n")


def simple_test(ManualMlp):
    """
    Simple focused test for debugging
    """
    print("\n" + "="*70)
    print("SIMPLE DEBUGGING TEST")
    print("="*70)
    
    test_suite = ManualMlpTestSuite()
    
    # Small test case
    in_features, hidden_features = 4, 16
    batch_size, seq_len = 1, 2
    
    print(f"\nSmall test: B={batch_size}, L={seq_len}, C={in_features}, Hidden={hidden_features}")
    
    # Create PyTorch Mlp
    mlp_pytorch = Mlp(in_features=in_features, hidden_features=hidden_features, 
                      out_features=in_features, drop=0.0)
    mlp_pytorch.eval()
    
    # Extract weights
    W1, B1, W2, B2 = test_suite.extract_pytorch_weights(mlp_pytorch)
    
    print(f"\nWeight shapes:")
    print(f"  W1: {W1.shape}, B1: {B1.shape}")
    print(f"  W2: {W2.shape}, B2: {B2.shape}")
    
    # Create manual version
    mlp_manual = ManualMlp(W1, B1, W2, B2, None)
    
    # Simple input
    x_torch = torch.randn(batch_size, seq_len, in_features)
    x_np = x_torch.numpy()
    
    print(f"\nInput shape: {x_torch.shape}")
    
    # Forward passes
    with torch.no_grad():
        out_torch = mlp_pytorch(x_torch).numpy()
    
    out_manual = mlp_manual(x_np.reshape(-1, in_features)).reshape(batch_size, seq_len, in_features)
    
    print(f"PyTorch output shape: {out_torch.shape}")
    print(f"Manual output shape: {out_manual.shape}")
    
    # Compare
    results = test_suite.compare_outputs(out_torch, out_manual, tolerance=1e-5,
                                        test_name="Simple debugging test")
    test_suite.print_results(results)
    
    print("="*70 + "\n")


# Example usage:
if __name__ == "__main__":
    # Import your ManualMlp implementation
    # from your_module import ManualMlp
    
    print("Test suite ready for comparing Original Swin Mlp with ManualMlp")
    print("\nAvailable test functions:")
    print("  1. simple_test(ManualMlp)      - Quick debugging test")
    print("  2. run_all_tests(ManualMlp)    - Comprehensive test suite")
    print("  3. test_edge_cases(ManualMlp)  - Edge case testing")
    print("\nExample usage:")
    print("  from your_module import ManualMlp")
    print("  simple_test(ManualMlp)")
    print("  run_all_tests(ManualMlp)")
    print("  test_edge_cases(ManualMlp)")

    simple_test(ManualMlp)
    run_all_tests(ManualMlp)
    test_edge_cases(ManualMlp)
