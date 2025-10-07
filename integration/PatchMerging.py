import numpy as np
import torch
import torch.nn as nn
from LayerNorm_v4 import LayerNorm
from linear import ExplicitLinear 
from typing import Dict, Tuple
import time
from original_swin import PatchMerging

class ManualPatchMerging:
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, reduction_weight=None, norm_weight=None, norm_bias=None, norm_layer=LayerNorm):
        self.input_resolution = input_resolution
        self.dim = dim

        self.reduction = ExplicitLinear(
            in_features=4 * dim, 
            out_features=2 * dim,
            weight=reduction_weight, 
            bias_condition=False
        )
        
        self.norm = norm_layer(
            normalized_shape=4 * dim, 
            weight=norm_weight, 
            bias=norm_bias
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # Convert to numpy if it's a tensor
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        H, W = self.input_resolution
        B, L, C = x_np.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x_np = x_np.reshape(B, H, W, C)

        x0 = x_np[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x_np[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x_np[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x_np[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x_np = np.concatenate([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x_np = x_np.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C

        x_np = self.norm(x_np)
        x_np = self.reduction(x_np)

        return x_np
    
    def __call__(self,x):
        return self.forward(x)

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
    


class PatchMergingTester:
    """
    Comprehensive test suite for comparing Manual NumPy and PyTorch PatchMerging implementations
    """
    
    def __init__(self, input_resolution=(56, 56), dim=96):
        """
        Initialize tester with configuration parameters
        
        Args:
            input_resolution: Input resolution tuple (H, W) - must be even numbers
            dim: Number of input channels
        """
        assert input_resolution[0] % 2 == 0 and input_resolution[1] % 2 == 0, \
            "input_resolution dimensions must be even numbers"
        
        self.input_resolution = input_resolution
        self.dim = dim
        
        # For reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def extract_pytorch_weights(self, pytorch_module: nn.Module) -> Dict:
        """
        Extract all weights from PyTorch PatchMerging
        
        Args:
            pytorch_module: PyTorch PatchMerging instance
            
        Returns:
            Dictionary containing all extracted weights as numpy arrays
        """
        weights = {}
        
        # Reduction layer (Linear) weights
        weights['reduction_weight'] = pytorch_module.reduction.weight.detach().cpu().numpy()
        # Note: reduction has bias=False, so no bias to extract
        
        # LayerNorm weights
        weights['norm_weight'] = pytorch_module.norm.weight.detach().cpu().numpy()
        weights['norm_bias'] = pytorch_module.norm.bias.detach().cpu().numpy()
        
        return weights
    
    def create_manual_patch_merging(self, weights: Dict):
        """
        Create manual PatchMerging with extracted weights
        
        Args:
            weights: Dictionary of extracted weights
            
        Returns:
            Manual PatchMerging instance
        """
        from LayerNorm_v4 import LayerNorm
        from linear import ExplicitLinear
        
        class ManualPatchMerging:
            """Pure NumPy implementation of PatchMerging"""
            
            def __init__(self, input_resolution, dim, reduction_weight, norm_weight, norm_bias):
                self.input_resolution = input_resolution
                self.dim = dim
                
                # Reduction layer
                self.reduction = ExplicitLinear(
                    in_features=4 * dim,
                    out_features=2 * dim,
                    weight=reduction_weight,
                    bias=None,
                    bias_condition=False
                )
                
                # Normalization layer
                self.norm = LayerNorm(
                    normalized_shape=4 * dim,
                    weight=norm_weight,
                    bias=norm_bias,
                    bias_condition=True
                )
            
            def forward(self, x):
                """
                x: B, H*W, C (numpy array)
                """
                # Convert to numpy if tensor
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                
                H, W = self.input_resolution
                B, L, C = x.shape
                assert L == H * W, f"input feature has wrong size: {L} != {H}*{W}"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                
                # Reshape to B, H, W, C
                x = x.reshape(B, H, W, C)
                
                # Extract patches (downsampling by 2)
                x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
                x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
                x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
                x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C
                
                # Concatenate along channel dimension
                x = np.concatenate([x0, x1, x2, x3], axis=-1)  # B, H/2, W/2, 4*C
                
                # Flatten spatial dimensions
                x = x.reshape(B, -1, 4 * C)  # B, H/2*W/2, 4*C
                
                # Normalize
                x = self.norm(x)
                
                # Reduce dimension
                x = self.reduction(x)
                
                return x
            
            def __call__(self, x):
                return self.forward(x)
        
        return ManualPatchMerging(
            input_resolution=self.input_resolution,
            dim=self.dim,
            reduction_weight=weights['reduction_weight'],
            norm_weight=weights['norm_weight'],
            norm_bias=weights['norm_bias']
        )
    
    def create_test_input(self, batch_size=2) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Create test input data
        
        Args:
            batch_size: Batch size for test input
            
        Returns:
            Tuple of (PyTorch tensor, NumPy array) with same values
        """
        H, W = self.input_resolution
        L = H * W
        
        # Create random input: B, H*W, C
        input_np = np.random.randn(batch_size, L, self.dim).astype(np.float32)
        input_torch = torch.from_numpy(input_np)
        
        return input_torch, input_np
    
    def test_output_shape(self, pytorch_module, manual_module, batch_size=2) -> Dict:
        """
        Test output shapes match expected dimensions
        
        Args:
            pytorch_module: PyTorch implementation
            manual_module: Manual NumPy implementation
            batch_size: Batch size for testing
            
        Returns:
            Dictionary with shape test results
        """
        results = {
            'passed': False,
            'pytorch_output_shape': None,
            'manual_output_shape': None,
            'expected_shape': None,
            'error': None
        }
        
        try:
            H, W = self.input_resolution
            expected_shape = (batch_size, (H // 2) * (W // 2), 2 * self.dim)
            results['expected_shape'] = expected_shape
            
            input_torch, input_np = self.create_test_input(batch_size)
            
            # PyTorch
            pytorch_module.eval()
            with torch.no_grad():
                output_torch = pytorch_module(input_torch)
            results['pytorch_output_shape'] = tuple(output_torch.shape)
            
            # Manual
            output_manual = manual_module(input_np)
            if isinstance(output_manual, torch.Tensor):
                output_manual = output_manual.detach().cpu().numpy()
            results['manual_output_shape'] = tuple(output_manual.shape)
            
            # Check shapes
            results['passed'] = (
                results['pytorch_output_shape'] == expected_shape and
                results['manual_output_shape'] == expected_shape
            )
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def test_forward_pass(self, pytorch_module, manual_module, 
                         batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test forward pass and compare outputs
        
        Args:
            pytorch_module: PyTorch implementation
            manual_module: Manual NumPy implementation
            batch_size: Batch size for testing
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': False,
            'max_diff': None,
            'mean_diff': None,
            'median_diff': None,
            'pytorch_output_shape': None,
            'manual_output_shape': None,
            'pytorch_time': None,
            'manual_time': None,
            'error': None
        }
        
        try:
            input_torch, input_np = self.create_test_input(batch_size)
            
            # PyTorch forward pass
            pytorch_module.eval()
            with torch.no_grad():
                start_time = time.time()
                output_torch = pytorch_module(input_torch)
                results['pytorch_time'] = time.time() - start_time
            
            output_torch_np = output_torch.detach().cpu().numpy()
            results['pytorch_output_shape'] = output_torch_np.shape
            
            # Manual forward pass
            start_time = time.time()
            output_manual = manual_module(input_np)
            results['manual_time'] = time.time() - start_time
            
            # Convert to numpy if needed
            if isinstance(output_manual, torch.Tensor):
                output_manual = output_manual.detach().cpu().numpy()
            
            results['manual_output_shape'] = output_manual.shape
            
            # Compare outputs
            diff = np.abs(output_torch_np - output_manual)
            results['max_diff'] = np.max(diff)
            results['mean_diff'] = np.mean(diff)
            results['median_diff'] = np.median(diff)
            results['passed'] = results['max_diff'] < tolerance
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def test_patch_extraction(self, pytorch_module, manual_module, batch_size=2) -> Dict:
        """
        Test that patches are extracted correctly (intermediate step)
        
        Args:
            pytorch_module: PyTorch implementation
            manual_module: Manual NumPy implementation
            batch_size: Batch size for testing
            
        Returns:
            Dictionary with patch extraction test results
        """
        results = {
            'passed': False,
            'error': None
        }
        
        try:
            input_torch, input_np = self.create_test_input(batch_size)
            H, W = self.input_resolution
            B, L, C = input_torch.shape
            
            # Manual patch extraction (NumPy)
            x_np = input_np.reshape(B, H, W, C)
            x0_np = x_np[:, 0::2, 0::2, :]
            x1_np = x_np[:, 1::2, 0::2, :]
            x2_np = x_np[:, 0::2, 1::2, :]
            x3_np = x_np[:, 1::2, 1::2, :]
            patches_np = np.concatenate([x0_np, x1_np, x2_np, x3_np], axis=-1)
            
            # PyTorch patch extraction
            x_torch = input_torch.view(B, H, W, C)
            x0_torch = x_torch[:, 0::2, 0::2, :]
            x1_torch = x_torch[:, 1::2, 0::2, :]
            x2_torch = x_torch[:, 0::2, 1::2, :]
            x3_torch = x_torch[:, 1::2, 1::2, :]
            patches_torch = torch.cat([x0_torch, x1_torch, x2_torch, x3_torch], dim=-1)
            patches_torch_np = patches_torch.detach().cpu().numpy()
            
            # Compare
            diff = np.abs(patches_torch_np - patches_np)
            results['max_diff'] = np.max(diff)
            results['mean_diff'] = np.mean(diff)
            results['passed'] = results['max_diff'] < 1e-6
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_multiple_batch_sizes(self, pytorch_module, manual_module, 
                                  batch_sizes=[1, 2, 4, 8]) -> Dict:
        """
        Test with different batch sizes
        
        Args:
            pytorch_module: PyTorch implementation
            manual_module: Manual NumPy implementation
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with results for each batch size
        """
        results = {}
        
        for batch_size in batch_sizes:
            try:
                input_torch, input_np = self.create_test_input(batch_size)
                
                # PyTorch
                with torch.no_grad():
                    output_torch = pytorch_module(input_torch)
                output_torch_np = output_torch.detach().cpu().numpy()
                
                # Manual
                output_manual = manual_module(input_np)
                if isinstance(output_manual, torch.Tensor):
                    output_manual = output_manual.detach().cpu().numpy()
                
                diff = np.abs(output_torch_np - output_manual)
                
                results[f'batch_{batch_size}'] = {
                    'pytorch_shape': tuple(output_torch_np.shape),
                    'manual_shape': tuple(output_manual.shape),
                    'max_diff': np.max(diff),
                    'mean_diff': np.mean(diff),
                    'passed': np.max(diff) < 1e-4,
                    'error': None
                }
            except Exception as e:
                results[f'batch_{batch_size}'] = {
                    'error': str(e)
                }
        
        return results
    
    def test_numerical_stability(self, pytorch_module, manual_module, 
                                num_runs=10) -> Dict:
        """
        Test numerical stability across multiple runs
        
        Args:
            pytorch_module: PyTorch implementation
            manual_module: Manual NumPy implementation
            num_runs: Number of test runs
            
        Returns:
            Dictionary with stability metrics
        """
        max_diffs = []
        mean_diffs = []
        errors = []
        
        for run in range(num_runs):
            try:
                input_torch, input_np = self.create_test_input(batch_size=2)
                
                with torch.no_grad():
                    output_torch = pytorch_module(input_torch)
                output_torch_np = output_torch.detach().cpu().numpy()
                
                output_manual = manual_module(input_np)
                if isinstance(output_manual, torch.Tensor):
                    output_manual = output_manual.detach().cpu().numpy()
                
                diff = np.abs(output_torch_np - output_manual)
                max_diffs.append(np.max(diff))
                mean_diffs.append(np.mean(diff))
            except Exception as e:
                errors.append(f"Run {run+1}: {str(e)}")
        
        if max_diffs:
            return {
                'max_diff_mean': np.mean(max_diffs),
                'max_diff_std': np.std(max_diffs),
                'max_diff_min': np.min(max_diffs),
                'max_diff_max': np.max(max_diffs),
                'mean_diff_mean': np.mean(mean_diffs),
                'mean_diff_std': np.std(mean_diffs),
                'successful_runs': len(max_diffs),
                'total_runs': num_runs,
                'all_passed': all(d < 1e-4 for d in max_diffs),
                'errors': errors if errors else None
            }
        else:
            return {
                'error': 'All runs failed',
                'errors': errors
            }
    
    def run_all_tests(self, pytorch_module, manual_module, verbose=True) -> Dict:
        """
        Run comprehensive test suite
        
        Args:
            pytorch_module: PyTorch implementation
            manual_module: Manual NumPy implementation
            verbose: Print detailed output
            
        Returns:
            Dictionary with all test results
        """
        if verbose:
            print("=" * 70)
            print("PATCH MERGING TEST SUITE")
            print("=" * 70)
            print(f"Configuration:")
            print(f"  input_resolution: {self.input_resolution}")
            print(f"  dim: {self.dim}")
            print(f"  output_resolution: ({self.input_resolution[0]//2}, {self.input_resolution[1]//2})")
            print(f"  output_dim: {2 * self.dim}")
            print("=" * 70)
        
        all_results = {}
        
        # Test 1: Output Shape
        if verbose:
            print("\n[Test 1] Output Shape Verification")
        shape_results = self.test_output_shape(pytorch_module, manual_module)
        all_results['output_shape'] = shape_results
        
        if verbose:
            if shape_results['error']:
                print(f"  ❌ ERROR: {shape_results['error']}")
            else:
                print(f"  ✓ Expected shape:  {shape_results['expected_shape']}")
                print(f"  ✓ PyTorch shape:   {shape_results['pytorch_output_shape']}")
                print(f"  ✓ Manual shape:    {shape_results['manual_output_shape']}")
                if shape_results['passed']:
                    print(f"  ✅ PASSED - Shapes match expected dimensions")
                else:
                    print(f"  ❌ FAILED - Shape mismatch")
        
        # Test 2: Forward Pass
        if verbose:
            print("\n[Test 2] Forward Pass Comparison")
        forward_results = self.test_forward_pass(pytorch_module, manual_module)
        all_results['forward_pass'] = forward_results
        
        if verbose:
            if forward_results['error']:
                print(f"  ❌ ERROR: {forward_results['error']}")
            else:
                print(f"  ✓ Max difference:    {forward_results['max_diff']:.2e}")
                print(f"  ✓ Mean difference:   {forward_results['mean_diff']:.2e}")
                print(f"  ✓ Median difference: {forward_results['median_diff']:.2e}")
                print(f"  ✓ PyTorch time:      {forward_results['pytorch_time']:.4f}s")
                print(f"  ✓ Manual time:       {forward_results['manual_time']:.4f}s")
                if forward_results['passed']:
                    print(f"  ✅ PASSED (tolerance: 1e-4)")
                else:
                    print(f"  ❌ FAILED (tolerance: 1e-4)")
        
        # Test 3: Patch Extraction
        if verbose:
            print("\n[Test 3] Patch Extraction Verification")
        patch_results = self.test_patch_extraction(pytorch_module, manual_module)
        all_results['patch_extraction'] = patch_results
        
        if verbose:
            if patch_results['error']:
                print(f"  ❌ ERROR: {patch_results['error']}")
            else:
                print(f"  ✓ Max difference:  {patch_results['max_diff']:.2e}")
                print(f"  ✓ Mean difference: {patch_results['mean_diff']:.2e}")
                if patch_results['passed']:
                    print(f"  ✅ PASSED - Patch extraction is correct")
                else:
                    print(f"  ❌ FAILED - Patch extraction differs")
        
        # Test 4: Multiple Batch Sizes
        if verbose:
            print("\n[Test 4] Multiple Batch Sizes")
        batch_results = self.test_multiple_batch_sizes(pytorch_module, manual_module)
        all_results['batch_sizes'] = batch_results
        
        if verbose:
            for batch_name, batch_result in batch_results.items():
                if 'error' in batch_result and batch_result['error']:
                    print(f"  ❌ {batch_name}: ERROR - {batch_result['error']}")
                else:
                    status = "✅" if batch_result['passed'] else "❌"
                    print(f"  {status} {batch_name}: max_diff={batch_result['max_diff']:.2e}")
        
        # Test 5: Numerical Stability
        if verbose:
            print("\n[Test 5] Numerical Stability (10 runs)")
        stability_results = self.test_numerical_stability(pytorch_module, manual_module)
        all_results['numerical_stability'] = stability_results
        
        if verbose:
            if 'error' in stability_results:
                print(f"  ❌ ERROR: {stability_results['error']}")
            else:
                print(f"  ✓ Successful runs:      {stability_results['successful_runs']}/{stability_results['total_runs']}")
                print(f"  ✓ Max diff (mean±std):  {stability_results['max_diff_mean']:.2e} ± {stability_results['max_diff_std']:.2e}")
                print(f"  ✓ Max diff (min-max):   {stability_results['max_diff_min']:.2e} - {stability_results['max_diff_max']:.2e}")
                print(f"  ✓ Mean diff (mean±std): {stability_results['mean_diff_mean']:.2e} ± {stability_results['mean_diff_std']:.2e}")
                if stability_results['all_passed']:
                    print(f"  ✅ PASSED - All runs within tolerance")
                else:
                    print(f"  ❌ FAILED - Some runs exceeded tolerance")
        
        # Summary
        if verbose:
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            
            tests_passed = 0
            total_tests = 0
            
            if not shape_results.get('error'):
                total_tests += 1
                if shape_results['passed']:
                    tests_passed += 1
                    print(f"  Output Shape:        ✅ PASS")
                else:
                    print(f"  Output Shape:        ❌ FAIL")
            
            if not forward_results.get('error'):
                total_tests += 1
                if forward_results['passed']:
                    tests_passed += 1
                    print(f"  Forward Pass:        ✅ PASS")
                else:
                    print(f"  Forward Pass:        ❌ FAIL")
            
            if not patch_results.get('error'):
                total_tests += 1
                if patch_results['passed']:
                    tests_passed += 1
                    print(f"  Patch Extraction:    ✅ PASS")
                else:
                    print(f"  Patch Extraction:    ❌ FAIL")
            
            all_batches_passed = all(
                r.get('passed', False) for r in batch_results.values() 
                if not r.get('error')
            )
            total_tests += 1
            if all_batches_passed:
                tests_passed += 1
                print(f"  Batch Sizes:         ✅ PASS")
            else:
                print(f"  Batch Sizes:         ❌ FAIL")
            
            if not stability_results.get('error'):
                total_tests += 1
                if stability_results['all_passed']:
                    tests_passed += 1
                    print(f"  Numerical Stability: ✅ PASS")
                else:
                    print(f"  Numerical Stability: ❌ FAIL")
            
            print(f"\n  Overall: {tests_passed}/{total_tests} tests passed")
            print("=" * 70)
        
        return all_results


# Example usage
def example_test_patch_merging():
    """
    Example: Test PatchMerging with standard configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE: PatchMerging Test")
    print("="*70)
    
    # Configuration
    input_resolution = (56, 56)
    dim = 96
    
    tester = PatchMergingTester(
        input_resolution=input_resolution,
        dim=dim
    )
    
    # Uncomment to run actual tests:
    
    # Create PyTorch module
    pytorch_module = PatchMerging(
        input_resolution=input_resolution,
        dim=dim,
        norm_layer=nn.LayerNorm
    )
    
    # Extract weights
    weights = tester.extract_pytorch_weights(pytorch_module)
    
    # Create manual module with same weights
    manual_module = tester.create_manual_patch_merging(weights)
    
    # Run all tests
    results = tester.run_all_tests(pytorch_module, manual_module)
    
    print("\n✓ Example setup complete. Uncomment code to run actual tests.")


if __name__ == "__main__":
    example_test_patch_merging()
    
    print("\n" + "="*70)
    print("To run actual tests:")
    print("1. Import your PatchMerging implementations")
    print("2. Create instances with same weights")
    print("3. Run tester.run_all_tests()")
    print("="*70)
