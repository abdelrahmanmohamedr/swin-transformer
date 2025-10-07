import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from timm.models.layers import DropPath, to_2tuple
import time

# Imports for original PyTorch implementation
from original_swin import SwinTransformerBlock

# Imports for manual implementations
from MLP import ManualMlp
from WindowAttention import WindowAttention  # Assuming you have this
from LayerNorm_v4 import LayerNorm
from GELU import GELU

WindowProcess = None
WindowProcessReverse = None

class ManualSwinTransformerBlock:
    """
    Manual NumPy-based implementation of Swin Transformer Block
    Accepts extracted weights from PyTorch version
    """
    def __init__(self, 
                 # Architecture parameters
                 dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 # Extracted weights
                 norm1_weight=None, norm1_bias=None,
                 attn_weights=None,  # Dictionary with attention weights
                 norm2_weight=None, norm2_bias=None,
                 mlp_fc1_weight=None, mlp_fc1_bias=None,
                 mlp_fc2_weight=None, mlp_fc2_bias=None,
                 attn_mask=None, qkv_bias_enabled=True, fused_window_process=False):
        """
        Args:
            dim: Number of input channels
            input_resolution: Input resolution tuple (H, W)
            num_heads: Number of attention heads
            window_size: Window size
            shift_size: Shift size for SW-MSA
            mlp_ratio: MLP hidden dimension ratio
            
            Weights extracted from PyTorch:
            norm1_weight, norm1_bias: LayerNorm 1 parameters
            attn_weights: Dictionary containing all attention module weights
            norm2_weight, norm2_bias: LayerNorm 2 parameters
            mlp_fc1_weight, mlp_fc1_bias: MLP first layer weights
            mlp_fc2_weight, mlp_fc2_bias: MLP second layer weights
            attn_mask: Attention mask for shifted windows
        """
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.window_partition = ManualMlp.window_partition
        self.window_reverse = ManualMlp.window_reverse
        
        # Adjust window size and shift size if needed
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        
        # LayerNorm 1
        self.norm1 = LayerNorm(dim, weight=norm1_weight, bias=norm1_bias, bias_condition=True)
        
        # Window Attention (you'll need to implement this to accept weights)
        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_weight=attn_weights['qkv_weight'],  # Pass all attention weights
            qkv_bias=attn_weights['qkv_bias'],  # Pass all attention weights
            proj_weight=attn_weights['proj_weight'],  # Pass all attention weights
            proj_bias=attn_weights['proj_bias'],  # Pass all attention weights
            qkv_bias_enabled=qkv_bias_enabled  # Pass all attention bias
        )
        
        # LayerNorm 2
        self.norm2 = LayerNorm(dim, weight=norm2_weight, bias=norm2_bias, bias_condition=True)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ManualMlp(mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias,
                             in_features=dim, hidden_features=mlp_hidden_dim, act_layer=GELU)
    
        # In SwinTransformerBlock.py, ManualSwinTransformerBlock.__init__()

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA using NumPy
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))  # Use numpy instead of torch.zeros

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)  # NumPy uses reshape, not view

            # Use NumPy broadcasting instead of unsqueeze
            attn_mask = mask_windows[:, np.newaxis, :] - mask_windows[:, :, np.newaxis]

            # Use np.where instead of masked_fill
            attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)

            # Convert to torch tensor for compatibility with attention operations
            attn_mask = torch.from_numpy(attn_mask)
        else:
            attn_mask = None

        self.attn_mask = attn_mask

        self.fused_window_process = fused_window_process

    def forward(self, x):
        """
        Forward pass that handles both PyTorch tensors and NumPy arrays
        """
        # Check input type and convert if necessary
        is_tensor_input = isinstance(x, torch.Tensor)
        if is_tensor_input:
            # Convert to numpy for processing
            device = x.device
            dtype = x.dtype
            x = x.detach().cpu().numpy()
        
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
    
        shortcut = x
        x = self.norm1(x)  # LayerNorm should handle both numpy and tensor
        
        # Ensure we have numpy array after norm
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        x = x.reshape(B, H, W, C)
    
        # cyclic shift - now x is definitely numpy
        if self.shift_size > 0:
            shifted_x = np.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
            x_windows = self.window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = self.window_partition(shifted_x, self.window_size)
    
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)
    
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
    
        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
    
        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
            x = np.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
    
        x = x.reshape(B, H * W, C)
        x = shortcut + x
    
        # FFN
        x = x + self.mlp(self.norm2(x))
    
        # Convert back to tensor if input was tensor
        if is_tensor_input:
            x = torch.from_numpy(x).to(device=device, dtype=dtype)
    
        return x

    def __call__(self,x):
        return self.forward(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

# Assuming imports from your modules
from original_swin import SwinTransformerBlock as PyTorchSwinBlock
from SwinTransformerBlock import ManualSwinTransformerBlock


class SwinTransformerBlockTester:
    """
    Comprehensive test suite for comparing Manual and PyTorch Swin Transformer implementations
    """
    
    def __init__(self, dim=96, input_resolution=(56, 56), num_heads=3, 
                 window_size=7, shift_size=0, mlp_ratio=4.0):
        """
        Initialize tester with configuration parameters
        
        Args:
            dim: Number of input channels
            input_resolution: Input resolution tuple (H, W)
            num_heads: Number of attention heads
            window_size: Window size for attention
            shift_size: Shift size for SW-MSA (0 for W-MSA, >0 for SW-MSA)
            mlp_ratio: MLP hidden dimension ratio
        """
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # For reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def extract_pytorch_weights(self, pytorch_block: nn.Module) -> Dict:
        """
        Extract all weights from PyTorch SwinTransformerBlock
        
        Args:
            pytorch_block: PyTorch SwinTransformerBlock instance
            
        Returns:
            Dictionary containing all extracted weights as numpy arrays
        """
        weights = {}
        
        # LayerNorm 1
        weights['norm1_weight'] = pytorch_block.norm1.weight.detach().cpu().numpy()
        weights['norm1_bias'] = pytorch_block.norm1.bias.detach().cpu().numpy()
        
        # Attention weights
        attn_weights = {}
        attn_weights['qkv_weight'] = pytorch_block.attn.qkv.weight.detach().cpu().numpy()
        attn_weights['qkv_bias'] = pytorch_block.attn.qkv.bias.detach().cpu().numpy()
        attn_weights['proj_weight'] = pytorch_block.attn.proj.weight.detach().cpu().numpy()
        attn_weights['proj_bias'] = pytorch_block.attn.proj.bias.detach().cpu().numpy()
        attn_weights['relative_position_bias_table'] = \
            pytorch_block.attn.relative_position_bias_table.detach().cpu().numpy()
        weights['attn_weights'] = attn_weights
        
        # LayerNorm 2
        weights['norm2_weight'] = pytorch_block.norm2.weight.detach().cpu().numpy()
        weights['norm2_bias'] = pytorch_block.norm2.bias.detach().cpu().numpy()
        
        # MLP weights
        weights['mlp_fc1_weight'] = pytorch_block.mlp.fc1.weight.detach().cpu().numpy()
        weights['mlp_fc1_bias'] = pytorch_block.mlp.fc1.bias.detach().cpu().numpy()
        weights['mlp_fc2_weight'] = pytorch_block.mlp.fc2.weight.detach().cpu().numpy()
        weights['mlp_fc2_bias'] = pytorch_block.mlp.fc2.bias.detach().cpu().numpy()
        
        # Attention mask
        if pytorch_block.attn_mask is not None:
            weights['attn_mask'] = pytorch_block.attn_mask.detach().cpu().numpy()
        else:
            weights['attn_mask'] = None
        
        return weights
    
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
        
        # Create random input
        input_np = np.random.randn(batch_size, L, self.dim).astype(np.float32)
        input_torch = torch.from_numpy(input_np)
        
        return input_torch, input_np
    
    def test_forward_pass(self, pytorch_block, manual_block, 
                         batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test forward pass and compare outputs
        
        Args:
            pytorch_block: PyTorch implementation
            manual_block: Manual NumPy implementation
            batch_size: Batch size for testing
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': False,
            'max_diff': None,
            'mean_diff': None,
            'pytorch_output_shape': None,
            'manual_output_shape': None,
            'pytorch_time': None,
            'manual_time': None,
            'error': None
        }
        
        try:
            # Create test input
            input_torch, input_np = self.create_test_input(batch_size)
            
            # PyTorch forward pass
            pytorch_block.eval()
            with torch.no_grad():
                start_time = time.time()
                output_torch = pytorch_block(input_torch)
                results['pytorch_time'] = time.time() - start_time
            
            output_torch_np = output_torch.detach().cpu().numpy()
            results['pytorch_output_shape'] = output_torch_np.shape
            
            # Manual forward pass
            start_time = time.time()
            output_manual = manual_block.forward(input_np)
            results['manual_time'] = time.time() - start_time
            
            # Convert manual output to numpy if it's a tensor
            if isinstance(output_manual, torch.Tensor):
                output_manual = output_manual.detach().cpu().numpy()
            
            results['manual_output_shape'] = output_manual.shape
            
            # Compare outputs
            diff = np.abs(output_torch_np - output_manual)
            results['max_diff'] = np.max(diff)
            results['mean_diff'] = np.mean(diff)
            results['passed'] = results['max_diff'] < tolerance
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def test_shape_consistency(self, pytorch_block, manual_block, 
                              batch_sizes=[1, 2, 4]) -> Dict:
        """
        Test output shapes for different batch sizes
        
        Args:
            pytorch_block: PyTorch implementation
            manual_block: Manual NumPy implementation
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with test results for each batch size
        """
        results = {}
        
        for batch_size in batch_sizes:
            try:
                input_torch, input_np = self.create_test_input(batch_size)
                
                # PyTorch
                with torch.no_grad():
                    output_torch = pytorch_block(input_torch)
                
                # Manual
                output_manual = manual_block.forward(input_np)
                
                # Convert to numpy if needed
                if isinstance(output_manual, torch.Tensor):
                    output_manual = output_manual.detach().cpu().numpy()
                
                results[f'batch_{batch_size}'] = {
                    'pytorch_shape': tuple(output_torch.shape),
                    'manual_shape': tuple(output_manual.shape),
                    'shapes_match': tuple(output_torch.shape) == tuple(output_manual.shape),
                    'error': None
                }
            except Exception as e:
                results[f'batch_{batch_size}'] = {
                    'error': str(e)
                }
        
        return results
    
    def test_numerical_stability(self, pytorch_block, manual_block, 
                                num_runs=5) -> Dict:
        """
        Test numerical stability across multiple runs
        
        Args:
            pytorch_block: PyTorch implementation
            manual_block: Manual NumPy implementation
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
                    output_torch = pytorch_block(input_torch)
                output_torch_np = output_torch.detach().cpu().numpy()
                
                output_manual = manual_block.forward(input_np)
                
                # Convert to numpy if needed
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
                'mean_diff_mean': np.mean(mean_diffs),
                'mean_diff_std': np.std(mean_diffs),
                'all_max_diffs': max_diffs,
                'all_mean_diffs': mean_diffs,
                'successful_runs': len(max_diffs),
                'total_runs': num_runs,
                'errors': errors if errors else None
            }
        else:
            return {
                'error': 'All runs failed',
                'errors': errors
            }
    
    def test_attention_mask(self, pytorch_block, manual_block) -> Dict:
        """
        Test attention mask computation for shifted windows
        
        Args:
            pytorch_block: PyTorch implementation
            manual_block: Manual NumPy implementation
            
        Returns:
            Dictionary with mask comparison results
        """
        results = {'mask_exists': False, 'masks_match': False}
        
        try:
            if pytorch_block.attn_mask is not None:
                results['mask_exists'] = True
                pytorch_mask = pytorch_block.attn_mask.detach().cpu().numpy()
                manual_mask = manual_block.attn_mask
                
                # Convert to numpy if needed
                if isinstance(manual_mask, torch.Tensor):
                    manual_mask = manual_mask.detach().cpu().numpy()
                
                if manual_mask is not None:
                    diff = np.abs(pytorch_mask - manual_mask)
                    results['max_diff'] = np.max(diff)
                    results['masks_match'] = results['max_diff'] < 1e-6
                    results['pytorch_mask_shape'] = pytorch_mask.shape
                    results['manual_mask_shape'] = manual_mask.shape
                else:
                    results['error'] = 'Manual mask is None but PyTorch mask exists'
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_individual_components(self, pytorch_block, manual_block, 
                                   batch_size=2) -> Dict:
        """
        Test individual components separately
        
        Args:
            pytorch_block: PyTorch implementation
            manual_block: Manual NumPy implementation
            batch_size: Batch size for testing
            
        Returns:
            Dictionary with component-wise test results
        """
        results = {}
        input_torch, input_np = self.create_test_input(batch_size)
        
        H, W = self.input_resolution
        B, L, C = input_torch.shape
        
        try:
            # Test LayerNorm 1
            pytorch_block.eval()
            with torch.no_grad():
                norm1_out_torch = pytorch_block.norm1(input_torch)
            norm1_out_torch_np = norm1_out_torch.detach().cpu().numpy()
            
            norm1_out_manual = manual_block.norm1(input_np)
            if isinstance(norm1_out_manual, torch.Tensor):
                norm1_out_manual = norm1_out_manual.detach().cpu().numpy()
            
            results['norm1'] = {
                'max_diff': np.max(np.abs(norm1_out_torch_np - norm1_out_manual)),
                'mean_diff': np.mean(np.abs(norm1_out_torch_np - norm1_out_manual)),
                'passed': np.max(np.abs(norm1_out_torch_np - norm1_out_manual)) < 1e-4
            }
        except Exception as e:
            results['norm1'] = {'error': str(e)}
        
        # You can add more component tests here (MLP, attention, etc.)
        
        return results
    
    def run_all_tests(self, pytorch_block, manual_block, verbose=True) -> Dict:
        """
        Run comprehensive test suite
        
        Args:
            pytorch_block: PyTorch implementation
            manual_block: Manual NumPy implementation
            verbose: Print detailed output
            
        Returns:
            Dictionary with all test results
        """
        if verbose:
            print("=" * 70)
            print("SWIN TRANSFORMER BLOCK TEST SUITE")
            print("=" * 70)
            print(f"Configuration:")
            print(f"  dim: {self.dim}")
            print(f"  input_resolution: {self.input_resolution}")
            print(f"  num_heads: {self.num_heads}")
            print(f"  window_size: {self.window_size}")
            print(f"  shift_size: {self.shift_size}")
            print(f"  mlp_ratio: {self.mlp_ratio}")
            print("=" * 70)
        
        all_results = {}
        
        # Test 1: Forward Pass
        if verbose:
            print("\n[Test 1] Forward Pass Comparison")
        forward_results = self.test_forward_pass(pytorch_block, manual_block)
        all_results['forward_pass'] = forward_results
        
        if forward_results['error']:
            if verbose:
                print(f"  ❌ ERROR: {forward_results['error']}")
                if 'traceback' in forward_results:
                    print(f"\nTraceback:\n{forward_results['traceback']}")
        else:
            if verbose:
                print(f"  ✓ PyTorch output shape: {forward_results['pytorch_output_shape']}")
                print(f"  ✓ Manual output shape:  {forward_results['manual_output_shape']}")
                print(f"  ✓ Max difference:       {forward_results['max_diff']:.2e}")
                print(f"  ✓ Mean difference:      {forward_results['mean_diff']:.2e}")
                print(f"  ✓ PyTorch time:         {forward_results['pytorch_time']:.4f}s")
                print(f"  ✓ Manual time:          {forward_results['manual_time']:.4f}s")
                
                if forward_results['passed']:
                    print(f"  ✅ PASSED (tolerance: 1e-4)")
                else:
                    print(f"  ❌ FAILED (tolerance: 1e-4)")
        
        # Test 2: Shape Consistency
        if verbose:
            print("\n[Test 2] Shape Consistency Across Batch Sizes")
        shape_results = self.test_shape_consistency(pytorch_block, manual_block)
        all_results['shape_consistency'] = shape_results
        
        if verbose:
            for batch_name, batch_result in shape_results.items():
                if 'error' in batch_result and batch_result['error']:
                    print(f"  ❌ {batch_name}: ERROR - {batch_result['error']}")
                else:
                    status = "✅" if batch_result['shapes_match'] else "❌"
                    print(f"  {status} {batch_name}: PyTorch{batch_result['pytorch_shape']} vs Manual{batch_result['manual_shape']}")
        
        # Test 3: Numerical Stability
        if verbose:
            print("\n[Test 3] Numerical Stability (5 runs)")
        stability_results = self.test_numerical_stability(pytorch_block, manual_block)
        all_results['numerical_stability'] = stability_results
        
        if verbose:
            if 'error' in stability_results:
                print(f"  ❌ ERROR: {stability_results['error']}")
                if stability_results.get('errors'):
                    for err in stability_results['errors']:
                        print(f"     - {err}")
            else:
                print(f"  ✓ Successful runs:  {stability_results['successful_runs']}/{stability_results['total_runs']}")
                print(f"  ✓ Max diff (mean):  {stability_results['max_diff_mean']:.2e}")
                print(f"  ✓ Max diff (std):   {stability_results['max_diff_std']:.2e}")
                print(f"  ✓ Mean diff (mean): {stability_results['mean_diff_mean']:.2e}")
                print(f"  ✓ Mean diff (std):  {stability_results['mean_diff_std']:.2e}")
        
        # Test 4: Attention Mask
        if verbose:
            print("\n[Test 4] Attention Mask Comparison")
        mask_results = self.test_attention_mask(pytorch_block, manual_block)
        all_results['attention_mask'] = mask_results
        
        if verbose:
            if 'error' in mask_results and mask_results['error']:
                print(f"  ❌ ERROR: {mask_results['error']}")
            elif mask_results['mask_exists']:
                if mask_results['masks_match']:
                    print(f"  ✅ Masks match (max diff: {mask_results['max_diff']:.2e})")
                else:
                    print(f"  ❌ Masks differ (max diff: {mask_results['max_diff']:.2e})")
                    print(f"     PyTorch shape: {mask_results.get('pytorch_mask_shape')}")
                    print(f"     Manual shape:  {mask_results.get('manual_mask_shape')}")
            else:
                print(f"  ✓ No attention mask (shift_size = 0)")
        
        # Test 5: Individual Components
        if verbose:
            print("\n[Test 5] Individual Component Tests")
        component_results = self.test_individual_components(pytorch_block, manual_block)
        all_results['components'] = component_results
        
        if verbose:
            for comp_name, comp_result in component_results.items():
                if 'error' in comp_result:
                    print(f"  ❌ {comp_name}: ERROR - {comp_result['error']}")
                else:
                    status = "✅" if comp_result['passed'] else "❌"
                    print(f"  {status} {comp_name}: max_diff={comp_result['max_diff']:.2e}, mean_diff={comp_result['mean_diff']:.2e}")
        
        # Summary
        if verbose:
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            
            if not forward_results.get('error'):
                forward_pass = "✅ PASS" if forward_results['passed'] else "❌ FAIL"
                print(f"  Forward Pass:        {forward_pass}")
            else:
                print(f"  Forward Pass:        ❌ ERROR")
            
            all_shapes_match = all(
                r.get('shapes_match', False) for r in shape_results.values() 
                if not r.get('error')
            )
            shape_test = "✅ PASS" if all_shapes_match else "❌ FAIL"
            print(f"  Shape Consistency:   {shape_test}")
            
            if not stability_results.get('error'):
                stability_ok = stability_results['max_diff_mean'] < 1e-4
                stability_test = "✅ PASS" if stability_ok else "❌ FAIL"
                print(f"  Numerical Stability: {stability_test}")
            else:
                print(f"  Numerical Stability: ❌ ERROR")
            
            if mask_results['mask_exists'] and not mask_results.get('error'):
                mask_test = "✅ PASS" if mask_results['masks_match'] else "❌ FAIL"
                print(f"  Attention Mask:      {mask_test}")
            
            print("=" * 70)
        
        return all_results


# Example usage and test scenarios
def example_test_w_msa():
    """
    Example: Test W-MSA (Window Multi-head Self Attention)
    shift_size = 0
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: W-MSA (shift_size=0)")
    print("="*70)
    
    tester = SwinTransformerBlockTester(
        dim=96,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        shift_size=0,  # W-MSA
        mlp_ratio=4.0
    )
    
    # Create PyTorch block
    pytorch_block = SwinTransformerBlock(
        dim=96,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        shift_size=0
    )
    
    # Extract weights
    weights = tester.extract_pytorch_weights(pytorch_block)
    
    # Create manual block with extracted weights
    manual_block = ManualSwinTransformerBlock(
        dim=96,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        shift_size=0,
        **weights
    )
    
    # Run tests
    results = tester.run_all_tests(pytorch_block, manual_block)
    
    print("\n✓ Example setup complete. Uncomment code to run actual tests.")


def example_test_sw_msa():
    """
    Example: Test SW-MSA (Shifted Window Multi-head Self Attention)
    shift_size > 0
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: SW-MSA (shift_size=3)")
    print("="*70)
    
    tester = SwinTransformerBlockTester(
        dim=96,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        shift_size=3,  # SW-MSA
        mlp_ratio=4.0
    )
    
    # Create PyTorch block
    pytorch_block = SwinTransformerBlock(
        dim=96,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        shift_size=3
    )
    
    # Extract weights
    weights = tester.extract_pytorch_weights(pytorch_block)
    
    # Create manual block with extracted weights
    manual_block = ManualSwinTransformerBlock(
        dim=96,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        shift_size=3,
        **weights
    )
    
    # Run tests
    results = tester.run_all_tests(pytorch_block, manual_block)
    print("\n✓ Example setup complete. Uncomment code to run actual tests.")


def example_test_different_configs():
    """
    Example: Test multiple configurations
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Configurations")
    print("="*70)
    
    configs = [
        {'dim': 96, 'input_resolution': (56, 56), 'num_heads': 3, 'window_size': 7, 'shift_size': 0},
        {'dim': 192, 'input_resolution': (28, 28), 'num_heads': 6, 'window_size': 7, 'shift_size': 3},
        {'dim': 384, 'input_resolution': (14, 14), 'num_heads': 12, 'window_size': 7, 'shift_size': 0},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"Configuration {i}: {config}")
        print('='*70)
        
        tester = SwinTransformerBlockTester(**config)

        # Create PyTorch block - USE CONFIG VALUES!
        pytorch_block = SwinTransformerBlock(**config)

        # Extract weights
        weights = tester.extract_pytorch_weights(pytorch_block)

        # Create manual block with extracted weights - USE CONFIG VALUES!
        manual_block = ManualSwinTransformerBlock(**config, **weights)

        # Run tests
        results = tester.run_all_tests(pytorch_block, manual_block)


if __name__ == "__main__":
    # Run example tests
    # example_test_w_msa()
    example_test_sw_msa()
    example_test_different_configs()
    
    print("\n" + "="*70)
    print("To run actual tests:")
    print("1. Import your implementations")
    print("2. Uncomment the block creation and testing code")
    print("3. Run the script")
    print("\nIMPORTANT: Your manual implementation should work entirely with NumPy")
    print("and avoid mixing PyTorch tensors with NumPy operations.")
    print("="*70)