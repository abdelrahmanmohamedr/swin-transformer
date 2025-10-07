import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Dict, Tuple, List
import time

# Note: You'll need to import these from your modules
# from SwinTransformerBlock import SwinTransformerBlock, ManualSwinTransformerBlock
# from LayerNorm_v4 import LayerNorm
# from ModuleList import ModuleList
# from original_swin import BasicLayer, PatchMerging
# from PatchMerging import ManualPatchMerging


class ManualBasicLayer:
    """A basic Swin Transformer layer for one stage - Fixed Version
    
    This version accepts per-block weights to match PyTorch's independent block weights.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., blocks_weights=None, downsample_weights=None,
                 norm_layer=None, downsample=None, use_checkpoint=False,
                 fused_window_process=False, qkv_bias=True):
        """
        Args:
            dim: Number of input channels
            input_resolution: Input resolution tuple (H, W)
            depth: Number of blocks
            num_heads: Number of attention heads
            window_size: Local window size
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            blocks_weights: List of weight dictionaries, one per block
            downsample_weights: Dictionary of downsample layer weights
            norm_layer: Normalization layer class
            downsample: Downsample layer class
            use_checkpoint: Whether to use checkpointing
            fused_window_process: Whether to use fused window processing
        """
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks with individual weights
        self.blocks = []
        if blocks_weights is None:
            raise ValueError("blocks_weights must be provided")
        
        if len(blocks_weights) != depth:
            raise ValueError(f"Expected {depth} block weights, got {len(blocks_weights)}")
        
        for i, block_w in enumerate(blocks_weights):
            # Import ManualSwinTransformerBlock from your module
            from SwinTransformerBlock import ManualSwinTransformerBlock
            
            block = ManualSwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=block_w['shift_size'],
                mlp_ratio=mlp_ratio,
                norm1_weight=block_w['norm1_weight'],
                norm1_bias=block_w['norm1_bias'],
                attn_weights=block_w['attn_weights'],
                norm2_weight=block_w['norm2_weight'],
                norm2_bias=block_w['norm2_bias'],
                mlp_fc1_weight=block_w['mlp_fc1_weight'],
                mlp_fc1_bias=block_w['mlp_fc1_bias'],
                mlp_fc2_weight=block_w['mlp_fc2_weight'],
                mlp_fc2_bias=block_w['mlp_fc2_bias'],
                attn_mask=block_w['attn_mask'],
                qkv_bias_enabled=qkv_bias,
                fused_window_process=fused_window_process
            )
            self.blocks.append(block)

        # Patch merging layer
        if downsample is not None:
            if downsample_weights is None:
                raise ValueError("downsample_weights must be provided when downsample is not None")
            
            self.downsample = downsample(
                input_resolution=input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                reduction_weight=downsample_weights['reduction_weight'],
                norm_weight=downsample_weights['norm_weight'],
                norm_bias=downsample_weights['norm_bias']
            )
        else:
            self.downsample = None

    def forward(self, x):
        """Forward pass through all blocks and optional downsample"""
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk.forward, x)
            else:
                x = blk.forward(x)
        if self.downsample is not None:
            x = self.downsample.forward(x)
        return x
    
    def __call__(self, x):
        """Make the layer callable"""
        return self.forward(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayerTester:
    """
    Comprehensive test suite for comparing Manual NumPy and PyTorch BasicLayer implementations
    """
    
    def __init__(self, dim=96, input_resolution=(56, 56), depth=2, 
                 num_heads=3, window_size=7, mlp_ratio=4.0):
        """
        Initialize tester with configuration parameters
        
        Args:
            dim: Number of input channels
            input_resolution: Input resolution tuple (H, W)
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            window_size: Window size for attention
            mlp_ratio: MLP hidden dimension ratio
        """
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        # For reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def extract_pytorch_weights(self, pytorch_layer) -> Dict:
        """
        Extract all weights from PyTorch BasicLayer
        
        Args:
            pytorch_layer: PyTorch BasicLayer instance
            
        Returns:
            Dictionary containing all extracted weights as numpy arrays
        """
        weights = {
            'blocks': [],
            'downsample': None
        }
        
        # Extract weights from each block separately
        for i, block in enumerate(pytorch_layer.blocks):
            block_weights = {
                # Store shift_size for this block
                'shift_size': block.shift_size,
                
                # LayerNorm 1
                'norm1_weight': block.norm1.weight.detach().cpu().numpy(),
                'norm1_bias': block.norm1.bias.detach().cpu().numpy(),
                
                # Attention weights
                'attn_weights': {
                    'qkv_weight': block.attn.qkv.weight.detach().cpu().numpy(),
                    'qkv_bias': block.attn.qkv.bias.detach().cpu().numpy() if block.attn.qkv.bias is not None else None,
                    'proj_weight': block.attn.proj.weight.detach().cpu().numpy(),
                    'proj_bias': block.attn.proj.bias.detach().cpu().numpy() if block.attn.proj.bias is not None else None,
                    'relative_position_bias_table': block.attn.relative_position_bias_table.detach().cpu().numpy()
                },
                
                # LayerNorm 2
                'norm2_weight': block.norm2.weight.detach().cpu().numpy(),
                'norm2_bias': block.norm2.bias.detach().cpu().numpy(),
                
                # MLP weights
                'mlp_fc1_weight': block.mlp.fc1.weight.detach().cpu().numpy(),
                'mlp_fc1_bias': block.mlp.fc1.bias.detach().cpu().numpy(),
                'mlp_fc2_weight': block.mlp.fc2.weight.detach().cpu().numpy(),
                'mlp_fc2_bias': block.mlp.fc2.bias.detach().cpu().numpy(),
                
                # Attention mask
                'attn_mask': block.attn_mask.detach().cpu().numpy() if block.attn_mask is not None else None
            }
            
            weights['blocks'].append(block_weights)
        
        # Extract downsample weights if present
        if pytorch_layer.downsample is not None:
            downsample_weights = {
                'reduction_weight': pytorch_layer.downsample.reduction.weight.detach().cpu().numpy(),
                'norm_weight': pytorch_layer.downsample.norm.weight.detach().cpu().numpy(),
                'norm_bias': pytorch_layer.downsample.norm.bias.detach().cpu().numpy()
            }
            weights['downsample'] = downsample_weights
        
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
    
    def test_forward_pass(self, pytorch_layer, manual_layer, 
                         batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test forward pass and compare outputs
        
        Args:
            pytorch_layer: PyTorch implementation
            manual_layer: Manual NumPy implementation
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
            # Create test input
            input_torch, input_np = self.create_test_input(batch_size)
            
            # PyTorch forward pass
            pytorch_layer.eval()
            with torch.no_grad():
                start_time = time.time()
                output_torch = pytorch_layer(input_torch)
                results['pytorch_time'] = time.time() - start_time
            
            output_torch_np = output_torch.detach().cpu().numpy()
            results['pytorch_output_shape'] = output_torch_np.shape
            
            # Manual forward pass
            start_time = time.time()
            output_manual = manual_layer.forward(input_np)
            results['manual_time'] = time.time() - start_time
            
            # Convert manual output to numpy if it's a tensor
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
    
    def test_individual_blocks(self, pytorch_layer, manual_layer, 
                              batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test each block individually
        
        Args:
            pytorch_layer: PyTorch implementation
            manual_layer: Manual NumPy implementation
            batch_size: Batch size for testing
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dictionary with per-block test results
        """
        results = {}
        
        try:
            input_torch, input_np = self.create_test_input(batch_size)
            
            # Test each block
            pytorch_layer.eval()
            x_torch = input_torch
            x_manual = input_np
            
            for i, (pytorch_block, manual_block) in enumerate(zip(
                pytorch_layer.blocks, manual_layer.blocks
            )):
                block_result = {
                    'shift_size': pytorch_block.shift_size,
                    'error': None
                }
                
                try:
                    # PyTorch block
                    with torch.no_grad():
                        out_torch = pytorch_block(x_torch)
                    out_torch_np = out_torch.detach().cpu().numpy()
                    
                    # Manual block
                    out_manual = manual_block.forward(x_manual)
                    if isinstance(out_manual, torch.Tensor):
                        out_manual = out_manual.detach().cpu().numpy()
                    
                    # Compare
                    diff = np.abs(out_torch_np - out_manual)
                    block_result['max_diff'] = np.max(diff)
                    block_result['mean_diff'] = np.mean(diff)
                    block_result['passed'] = block_result['max_diff'] < tolerance
                    
                    # Update for next block
                    x_torch = out_torch
                    x_manual = out_manual
                    
                except Exception as e:
                    block_result['error'] = str(e)
                    import traceback
                    block_result['traceback'] = traceback.format_exc()
                
                results[f'block_{i}'] = block_result
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def test_downsample(self, pytorch_layer, manual_layer, 
                       batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test downsample layer if present
        
        Args:
            pytorch_layer: PyTorch implementation
            manual_layer: Manual NumPy implementation
            batch_size: Batch size for testing
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dictionary with downsample test results
        """
        results = {
            'has_downsample': False,
            'passed': False,
            'error': None
        }
        
        if pytorch_layer.downsample is None:
            results['message'] = 'No downsample layer present'
            return results
        
        results['has_downsample'] = True
        
        try:
            # Get intermediate output (after all blocks, before downsample)
            input_torch, input_np = self.create_test_input(batch_size)
            
            pytorch_layer.eval()
            with torch.no_grad():
                x_torch = input_torch
                for block in pytorch_layer.blocks:
                    x_torch = block(x_torch)
                
                # Apply downsample
                out_torch = pytorch_layer.downsample(x_torch)
            
            # Manual version
            x_manual = input_np
            for block in manual_layer.blocks:
                x_manual = block.forward(x_manual)
            
            out_manual = manual_layer.downsample.forward(x_manual)
            
            # Convert to numpy
            out_torch_np = out_torch.detach().cpu().numpy()
            if isinstance(out_manual, torch.Tensor):
                out_manual = out_manual.detach().cpu().numpy()
            
            # Compare
            diff = np.abs(out_torch_np - out_manual)
            results['max_diff'] = np.max(diff)
            results['mean_diff'] = np.mean(diff)
            results['output_shape_pytorch'] = out_torch_np.shape
            results['output_shape_manual'] = out_manual.shape
            results['passed'] = results['max_diff'] < tolerance
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def test_shift_pattern(self, pytorch_layer, manual_layer) -> Dict:
        """
        Verify that blocks alternate between W-MSA and SW-MSA correctly
        
        Returns:
            Dictionary with shift pattern verification
        """
        results = {
            'pytorch_pattern': [],
            'manual_pattern': [],
            'patterns_match': False,
            'expected_pattern': []
        }
        
        try:
            # Expected pattern: even indices = 0, odd indices = window_size // 2
            expected_pattern = [
                0 if i % 2 == 0 else self.window_size // 2 
                for i in range(self.depth)
            ]
            results['expected_pattern'] = expected_pattern
            
            # PyTorch pattern
            pytorch_pattern = [block.shift_size for block in pytorch_layer.blocks]
            results['pytorch_pattern'] = pytorch_pattern
            
            # Manual pattern
            manual_pattern = [block.shift_size for block in manual_layer.blocks]
            results['manual_pattern'] = manual_pattern
            
            # Verify
            results['patterns_match'] = (
                pytorch_pattern == expected_pattern and
                manual_pattern == expected_pattern
            )
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_numerical_stability(self, pytorch_layer, manual_layer, 
                                num_runs=5) -> Dict:
        """
        Test numerical stability across multiple runs
        
        Args:
            pytorch_layer: PyTorch implementation
            manual_layer: Manual NumPy implementation
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
                
                pytorch_layer.eval()
                with torch.no_grad():
                    output_torch = pytorch_layer(input_torch)
                output_torch_np = output_torch.detach().cpu().numpy()
                
                output_manual = manual_layer.forward(input_np)
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
    
    def run_all_tests(self, pytorch_layer, manual_layer, verbose=True) -> Dict:
        """
        Run comprehensive test suite
        
        Args:
            pytorch_layer: PyTorch implementation
            manual_layer: Manual NumPy implementation
            verbose: Print detailed output
            
        Returns:
            Dictionary with all test results
        """
        if verbose:
            print("=" * 70)
            print("BASIC LAYER TEST SUITE")
            print("=" * 70)
            print(f"Configuration:")
            print(f"  dim: {self.dim}")
            print(f"  input_resolution: {self.input_resolution}")
            print(f"  depth: {self.depth}")
            print(f"  num_heads: {self.num_heads}")
            print(f"  window_size: {self.window_size}")
            print(f"  mlp_ratio: {self.mlp_ratio}")
            print(f"  has_downsample: {pytorch_layer.downsample is not None}")
            print("=" * 70)
        
        all_results = {}
        
        # Test 1: Shift Pattern
        if verbose:
            print("\n[Test 1] Shift Pattern Verification")
        shift_results = self.test_shift_pattern(pytorch_layer, manual_layer)
        all_results['shift_pattern'] = shift_results
        
        if verbose:
            if 'error' in shift_results:
                print(f"  ERROR: {shift_results['error']}")
            else:
                print(f"  Expected pattern:  {shift_results['expected_pattern']}")
                print(f"  PyTorch pattern:   {shift_results['pytorch_pattern']}")
                print(f"  Manual pattern:    {shift_results['manual_pattern']}")
                if shift_results['patterns_match']:
                    print(f"  PASSED - Patterns match (W-MSA/SW-MSA alternation)")
                else:
                    print(f"  FAILED - Pattern mismatch")
        
        # Test 2: Individual Blocks
        if verbose:
            print("\n[Test 2] Individual Block Tests")
        block_results = self.test_individual_blocks(pytorch_layer, manual_layer)
        all_results['individual_blocks'] = block_results
        
        if verbose:
            if 'error' in block_results:
                print(f"  ERROR: {block_results['error']}")
                if 'traceback' in block_results:
                    print(f"\nTraceback:\n{block_results['traceback']}")
            else:
                for block_name, block_result in block_results.items():
                    if 'error' in block_result and block_result['error']:
                        print(f"  {block_name}: ERROR - {block_result['error']}")
                    else:
                        status = "PASS" if block_result['passed'] else "FAIL"
                        shift_type = "W-MSA" if block_result['shift_size'] == 0 else "SW-MSA"
                        print(f"  {status}: {block_name} ({shift_type}): "
                              f"max_diff={block_result['max_diff']:.2e}")
        
        # Test 3: Forward Pass
        if verbose:
            print("\n[Test 3] Full Forward Pass")
        forward_results = self.test_forward_pass(pytorch_layer, manual_layer)
        all_results['forward_pass'] = forward_results
        
        if verbose:
            if forward_results['error']:
                print(f"  ERROR: {forward_results['error']}")
                if 'traceback' in forward_results:
                    print(f"\nTraceback:\n{forward_results['traceback']}")
            else:
                print(f"  PyTorch output shape: {forward_results['pytorch_output_shape']}")
                print(f"  Manual output shape:  {forward_results['manual_output_shape']}")
                print(f"  Max difference:       {forward_results['max_diff']:.2e}")
                print(f"  Mean difference:      {forward_results['mean_diff']:.2e}")
                print(f"  Median difference:    {forward_results['median_diff']:.2e}")
                print(f"  PyTorch time:         {forward_results['pytorch_time']:.4f}s")
                print(f"  Manual time:          {forward_results['manual_time']:.4f}s")
                
                if forward_results['passed']:
                    print(f"  PASSED (tolerance: 1e-4)")
                else:
                    print(f"  FAILED (tolerance: 1e-4)")
        
        # Test 4: Downsample
        if verbose:
            print("\n[Test 4] Downsample Layer")
        downsample_results = self.test_downsample(pytorch_layer, manual_layer)
        all_results['downsample'] = downsample_results
        
        if verbose:
            if not downsample_results['has_downsample']:
                print(f"  No downsample layer (as expected)")
            elif downsample_results['error']:
                print(f"  ERROR: {downsample_results['error']}")
                if 'traceback' in downsample_results:
                    print(f"\nTraceback:\n{downsample_results['traceback']}")
            else:
                print(f"  Output shape (PyTorch): {downsample_results['output_shape_pytorch']}")
                print(f"  Output shape (Manual):  {downsample_results['output_shape_manual']}")
                print(f"  Max difference:         {downsample_results['max_diff']:.2e}")
                print(f"  Mean difference:        {downsample_results['mean_diff']:.2e}")
                if downsample_results['passed']:
                    print(f"  PASSED")
                else:
                    print(f"  FAILED")
        
        # Test 5: Numerical Stability
        if verbose:
            print("\n[Test 5] Numerical Stability (5 runs)")
        stability_results = self.test_numerical_stability(pytorch_layer, manual_layer)
        all_results['numerical_stability'] = stability_results
        
        if verbose:
            if 'error' in stability_results:
                print(f"  ERROR: {stability_results['error']}")
            else:
                print(f"  Successful runs:      {stability_results['successful_runs']}/{stability_results['total_runs']}")
                print(f"  Max diff (mean±std):  {stability_results['max_diff_mean']:.2e} ± {stability_results['max_diff_std']:.2e}")
                print(f"  Max diff (min-max):   {stability_results['max_diff_min']:.2e} - {stability_results['max_diff_max']:.2e}")
                print(f"  Mean diff (mean±std): {stability_results['mean_diff_mean']:.2e} ± {stability_results['mean_diff_std']:.2e}")
                if stability_results['all_passed']:
                    print(f"  PASSED - All runs within tolerance")
                else:
                    print(f"  FAILED - Some runs exceeded tolerance")
        
        # Summary
        if verbose:
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            
            tests_passed = 0
            total_tests = 0
            
            # Shift pattern
            if not shift_results.get('error'):
                total_tests += 1
                if shift_results['patterns_match']:
                    tests_passed += 1
                    print(f"  Shift Pattern:       PASS")
                else:
                    print(f"  Shift Pattern:       FAIL")
            
            # Individual blocks
            if not block_results.get('error'):
                all_blocks_passed = all(
                    r.get('passed', False) for r in block_results.values()
                    if not r.get('error')
                )
                total_tests += 1
                if all_blocks_passed:
                    tests_passed += 1
                    print(f"  Individual Blocks:   PASS")
                else:
                    print(f"  Individual Blocks:   FAIL")
            
            # Forward pass
            if not forward_results.get('error'):
                total_tests += 1
                if forward_results['passed']:
                    tests_passed += 1
                    print(f"  Forward Pass:        PASS")
                else:
                    print(f"  Forward Pass:        FAIL")
            
            # Downsample
            if downsample_results['has_downsample'] and not downsample_results.get('error'):
                total_tests += 1
                if downsample_results['passed']:
                    tests_passed += 1
                    print(f"  Downsample:          PASS")
                else:
                    print(f"  Downsample:          FAIL")
            
            # Stability
            if not stability_results.get('error'):
                total_tests += 1
                if stability_results['all_passed']:
                    tests_passed += 1
                    print(f"  Numerical Stability: PASS")
                else:
                    print(f"  Numerical Stability: FAIL")
            
            print(f"\n  Overall: {tests_passed}/{total_tests} tests passed")
            print("=" * 70)
        
        return all_results


def example_test_basic_layer():
    """
    Example: Test BasicLayer with and without downsample
    
    NOTE: This function requires proper imports. Make sure you have:
    - original_swin.BasicLayer
    - PatchMerging.ManualPatchMerging
    - LayerNorm_v4.LayerNorm
    - SwinTransformerBlock.ManualSwinTransformerBlock
    """
    # Import your modules
    from original_swin import BasicLayer, PatchMerging
    from PatchMerging import ManualPatchMerging
    from LayerNorm_v4 import LayerNorm
    
    print("\n" + "="*70)
    print("EXAMPLE: BasicLayer Test")
    print("="*70)
    
    # Test Configuration 1: Without Downsample
    print("\n--- Configuration 1: Without Downsample ---")
    tester1 = BasicLayerTester(
        dim=96,
        input_resolution=(56, 56),
        depth=2,
        num_heads=3,
        window_size=7,
        mlp_ratio=4.0
    )
    
    # Create PyTorch layer
    pytorch_layer1 = BasicLayer(
        dim=96,
        input_resolution=(56, 56),
        depth=2,
        num_heads=3,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        downsample=None
    )
    
    # Extract weights
    weights1 = tester1.extract_pytorch_weights(pytorch_layer1)
    
    # Create manual layer with extracted weights
    manual_layer1 = ManualBasicLayer(
        dim=96,
        input_resolution=(56, 56),
        depth=2,
        num_heads=3,
        window_size=7,
        mlp_ratio=4.0,
        blocks_weights=weights1['blocks'],
        downsample_weights=None,
        norm_layer=LayerNorm,
        downsample=None,
        use_checkpoint=False
    )
    
    # Run tests
    results1 = tester1.run_all_tests(pytorch_layer1, manual_layer1)
    
    # Test Configuration 2: With PatchMerging Downsample
    print("\n--- Configuration 2: With PatchMerging Downsample ---")
    tester2 = BasicLayerTester(
        dim=96,
        input_resolution=(56, 56),
        depth=2,
        num_heads=3,
        window_size=7,
        mlp_ratio=4.0
    )
    
    # Create PyTorch layer with downsample
    pytorch_layer2 = BasicLayer(
        dim=96,
        input_resolution=(56, 56),
        depth=2,
        num_heads=3,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        downsample=PatchMerging
    )
    
    # Extract weights
    weights2 = tester2.extract_pytorch_weights(pytorch_layer2)
    
    # Create manual layer with extracted weights
    manual_layer2 = ManualBasicLayer(
        dim=96,
        input_resolution=(56, 56),
        depth=2,
        num_heads=3,
        window_size=7,
        mlp_ratio=4.0,
        blocks_weights=weights2['blocks'],
        downsample_weights=weights2['downsample'],
        norm_layer=LayerNorm,
        downsample=ManualPatchMerging,
        use_checkpoint=False
    )
    
    # Run tests
    results2 = tester2.run_all_tests(pytorch_layer2, manual_layer2)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nKey Points:")
    print("  1. Each block has independent weights (not shared)")
    print("  2. Blocks alternate W-MSA (shift=0) and SW-MSA (shift=window_size//2)")
    print("  3. Optional PatchMerging downsample reduces spatial resolution by 2x")
    print("  4. All components tested individually and as a complete layer")
    print("="*70)


if __name__ == "__main__":
    example_test_basic_layer()
    
    print("\n" + "="*70)
    print("BasicLayer Architecture Summary:")
    print("="*70)
    print("Structure:")
    print("  - Input: [B, H*W, C]")
    print("  - Multiple SwinTransformerBlocks (depth parameter)")
    print("    - Block 0: W-MSA (shift_size=0)")
    print("    - Block 1: SW-MSA (shift_size=window_size//2)")
    print("    - Block 2: W-MSA (shift_size=0)")
    print("    - ... (alternating pattern)")
    print("  - Optional PatchMerging downsample")
    print("  - Output: [B, (H/2)*(W/2), 2*C] if downsample, else [B, H*W, C]")
    print()
    print("Testing Features:")
    print("  - Weight extraction from PyTorch")
    print("  - Per-block weight initialization")
    print("  - Shift pattern verification")
    print("  - Individual block testing")
    print("  - Full forward pass comparison")
    print("  - Downsample layer testing")
    print("  - Numerical stability across runs")
    print("="*70)