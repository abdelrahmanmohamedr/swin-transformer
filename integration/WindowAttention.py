import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from linear import ExplicitLinear
from softmax import Softmax

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qkv_bias (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, 
             qkv_weight=None, qkv_bias=None,
             proj_weight=None, proj_bias=None,
             qkv_bias_enabled=True, qk_scale=None, 
             attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Replace nn.Linear with ExplicitLinear
        self.qkv = ExplicitLinear(in_features=dim, out_features= dim * 3,
                                          weight=qkv_weight, bias=qkv_bias, bias_condition=qkv_bias_enabled)
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = ExplicitLinear(in_features=dim, out_features= dim,
                                          weight=proj_weight, bias=proj_bias, bias_condition=True)
        
        self.proj_drop = nn.Dropout(proj_drop)


        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Convert to numpy for ExplicitLinear, then back to torch
        x_np = x.detach().cpu().numpy()
        qkv_np = self.qkv(x_np)
        qkv = torch.from_numpy(qkv_np).to(x.device).float()
        
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        if isinstance(attn, np.ndarray):
            attn = torch.from_numpy(attn).to(x.device)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Convert to numpy for ExplicitLinear, then back to torch
        x_np = x.detach().cpu().numpy()
        x_proj_np = self.proj(x_np)
        x = torch.from_numpy(x_proj_np).to(x.device).float()
        
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
# ==========================================================================================================
# TEST HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------------------------
def copy_weights(impl_module, api_module):

    """
    Copy weights from API module to implemented module to ensure fair comparison.
    
    Args:
        impl_module: Your implemented WindowAttention
        api_module: Original API WindowAttention
    """
    with torch.no_grad():
        # Copy QKV weights
        if hasattr(impl_module.qkv, 'weight'):
            api_module.qkv.weight.copy_(torch.from_numpy(impl_module.qkv.weight))
        if hasattr(impl_module.qkv, 'bias') and api_module.qkv.bias is not None:
            api_module.qkv.bias.copy_(torch.from_numpy(impl_module.qkv.bias))
        
        # Copy projection weights
        if hasattr(impl_module.proj, 'weight'):
            api_module.proj.weight.copy_(torch.from_numpy(impl_module.proj.weight))
        if hasattr(impl_module.proj, 'bias') and api_module.proj.bias is not None:
            api_module.proj.bias.copy_(torch.from_numpy(impl_module.proj.bias))

        
        # Copy relative position bias table
        impl_module.relative_position_bias_table.copy_(api_module.relative_position_bias_table)


def create_test_input(num_windows, window_size, dim, device='cpu'):
    """
    Create test input for WindowAttention.
    
    Args:
        num_windows: Number of windows (typically num_windows * batch_size)
        window_size: Tuple (Wh, Ww)
        dim: Channel dimension
        device: 'cpu' or 'cuda'
    
    Returns:
        torch.Tensor of shape (num_windows, Wh*Ww, dim)
    """
    N = window_size[0] * window_size[1]
    x = torch.randn(num_windows, N, dim, device=device)
    return x


def create_attention_mask(num_windows, window_size, shift_size):
    """
    Create attention mask for shifted window attention.
    
    Args:
        num_windows: Number of windows
        window_size: Window size
        shift_size: Shift size for SW-MSA
    
    Returns:
        Attention mask or None
    """
    if shift_size == 0:
        return None
    
    # Simplified mask creation for testing
    N = window_size * window_size
    mask = torch.zeros(num_windows, N, N)
    return mask
# ==========================================================================================================

# ==========================================================================================================
# TEST CASES
# ----------------------------------------------------------------------------------------------------------
def test_initialization(test_name=""):
    """Test that both modules initialize with same structure"""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    dim = 96
    window_size = (7, 7)
    num_heads = 3
    
    try:
        # Import your modules
        from WindowAttention import WindowAttention as WindowAttentionImpl
        from original_swin import WindowAttention as WindowAttentionAPI
        
        # Create instances
        impl_attn = WindowAttentionImpl(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads
        )
        
        api_attn = WindowAttentionAPI(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads
        )
        
        print(f"✓ Both modules created successfully")
        print(f"\nImplemented WindowAttention:")
        print(f"  - dim: {impl_attn.dim}")
        print(f"  - window_size: {impl_attn.window_size}")
        print(f"  - num_heads: {impl_attn.num_heads}")
        print(f"  - scale: {impl_attn.scale:.6f}")
        
        print(f"\nAPI WindowAttention:")
        print(f"  - dim: {api_attn.dim}")
        print(f"  - window_size: {api_attn.window_size}")
        print(f"  - num_heads: {api_attn.num_heads}")
        print(f"  - scale: {api_attn.scale:.6f}")
        
        # Check structure
        structure_match = (
            impl_attn.dim == api_attn.dim and
            impl_attn.window_size == api_attn.window_size and
            impl_attn.num_heads == api_attn.num_heads and
            abs(impl_attn.scale - api_attn.scale) < 1e-6
        )
        
        print(f"\n✓ Structure matches: {structure_match}")
        
        # Check relative position bias table shape
        impl_bias_shape = impl_attn.relative_position_bias_table.shape
        api_bias_shape = api_attn.relative_position_bias_table.shape
        print(f"\nRelative position bias table shapes:")
        print(f"  - Implemented: {impl_bias_shape}")
        print(f"  - API: {api_bias_shape}")
        print(f"  - Match: {impl_bias_shape == api_bias_shape}")
        
        if structure_match:
            print("\n✓ PASSED")
        else:
            print("\n✗ FAILED")
        
        return structure_match
        
    except Exception as e:
        print(f"\n✗ FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_no_mask(test_name=""):
    """Test forward pass without attention mask"""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    try:
        from WindowAttention import WindowAttention as WindowAttentionImpl
        from original_swin import WindowAttention as WindowAttentionAPI
        
        # Configuration
        dim = 96
        window_size = (7, 7)
        num_heads = 3
        num_windows = 64  # 8x8 = 64 windows for a 56x56 image with 7x7 windows
        
        # Create modules
        torch.manual_seed(42)
        impl_attn = WindowAttentionImpl(dim=dim, window_size=window_size, num_heads=num_heads)
        
        torch.manual_seed(42)
        api_attn = WindowAttentionAPI(dim=dim, window_size=window_size, num_heads=num_heads)
        
        # Copy weights to ensure same initialization
        copy_weights(impl_attn, api_attn)
        
        # Create input
        torch.manual_seed(123)
        x = create_test_input(num_windows, window_size, dim)
        
        # Forward pass
        impl_attn.eval()
        api_attn.eval()
        
        with torch.no_grad():
            impl_output = impl_attn(x.clone())
            api_output = api_attn(x.clone())
        
        print(f"Input shape: {x.shape}")
        print(f"Implemented output shape: {impl_output.shape}")
        print(f"API output shape: {api_output.shape}")
        
        # Compare outputs
        max_diff = torch.abs(impl_output - api_output).max().item()
        mean_diff = torch.abs(impl_output - api_output).mean().item()
        
        print(f"\nOutput comparison:")
        print(f"  - Max difference: {max_diff:.10f}")
        print(f"  - Mean difference: {mean_diff:.10f}")
        
        # Check if outputs are close
        match = torch.allclose(impl_output, api_output, atol=1e-5, rtol=1e-4)
        print(f"  - Outputs match (atol=1e-5, rtol=1e-4): {match}")
        
        # Sample values
        print(f"\nSample values (first 5 elements):")
        print(f"  - Implemented: {impl_output[0, 0, :5].tolist()}")
        print(f"  - API: {api_output[0, 0, :5].tolist()}")
        
        if match:
            print("\n✓ PASSED")
        else:
            print("\n✗ FAILED - Outputs don't match closely enough")
            print("  This could be due to numerical precision differences")
            print("  in ExplicitLinear vs nn.Linear implementations")
        
        return match
        
    except Exception as e:
        print(f"\n✗ FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_with_mask(test_name=""):
    """Test forward pass with attention mask (shifted window)"""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    try:
        from WindowAttention import WindowAttention as WindowAttentionImpl
        from original_swin import WindowAttention as WindowAttentionAPI
        
        # Configuration
        dim = 96
        window_size = (7, 7)
        num_heads = 3
        num_windows = 64
        
        # Create modules
        torch.manual_seed(42)
        impl_attn = WindowAttentionImpl(dim=dim, window_size=window_size, num_heads=num_heads)
        
        torch.manual_seed(42)
        api_attn = WindowAttentionAPI(dim=dim, window_size=window_size, num_heads=num_heads)
        
        # Copy weights
        copy_weights(impl_attn, api_attn)
        
        # Create input and mask
        torch.manual_seed(123)
        x = create_test_input(num_windows, window_size, dim)
        
        # Create a simple mask
        N = window_size[0] * window_size[1]
        mask = torch.zeros(num_windows, N, N)
        # Set some entries to -100 to mask them
        mask[:, :10, :10] = -100.0
        
        # Forward pass
        impl_attn.eval()
        api_attn.eval()
        
        with torch.no_grad():
            impl_output = impl_attn(x.clone(), mask=mask.clone())
            api_output = api_attn(x.clone(), mask=mask.clone())
        
        print(f"Input shape: {x.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Implemented output shape: {impl_output.shape}")
        print(f"API output shape: {api_output.shape}")
        
        # Compare outputs
        max_diff = torch.abs(impl_output - api_output).max().item()
        mean_diff = torch.abs(impl_output - api_output).mean().item()
        
        print(f"\nOutput comparison:")
        print(f"  - Max difference: {max_diff:.10f}")
        print(f"  - Mean difference: {mean_diff:.10f}")
        
        match = torch.allclose(impl_output, api_output, atol=1e-5, rtol=1e-4)
        print(f"  - Outputs match (atol=1e-5, rtol=1e-4): {match}")
        
        if match:
            print("\n✓ PASSED")
        else:
            print("\n✗ FAILED - Outputs don't match with mask")
        
        return match
        
    except Exception as e:
        print(f"\n✗ FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_flops_calculation(test_name=""):
    """Test FLOPS calculation (should be identical)"""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    try:
        from WindowAttention import WindowAttention as WindowAttentionImpl
        from original_swin import WindowAttention as WindowAttentionAPI
        
        dim = 96
        window_size = (7, 7)
        num_heads = 3
        N = window_size[0] * window_size[1]  # 49
        
        impl_attn = WindowAttentionImpl(dim=dim, window_size=window_size, num_heads=num_heads)
        api_attn = WindowAttentionAPI(dim=dim, window_size=window_size, num_heads=num_heads)
        
        impl_flops = impl_attn.flops(N)
        api_flops = api_attn.flops(N)
        
        print(f"Window size: {window_size}, N={N}")
        print(f"Implemented FLOPS: {impl_flops:,}")
        print(f"API FLOPS: {api_flops:,}")
        print(f"Difference: {abs(impl_flops - api_flops):,}")
        
        match = impl_flops == api_flops
        
        if match:
            print("\n✓ PASSED")
        else:
            print("\n✗ FAILED")
        
        return match
        
    except Exception as e:
        print(f"\n✗ FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_relative_position_bias(test_name=""):
    """Test relative position bias computation"""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    try:
        from WindowAttention import WindowAttention as WindowAttentionImpl
        from original_swin import WindowAttention as WindowAttentionAPI
        
        dim = 96
        window_size = (7, 7)
        num_heads = 3
        
        torch.manual_seed(42)
        impl_attn = WindowAttentionImpl(dim=dim, window_size=window_size, num_heads=num_heads)
        
        torch.manual_seed(42)
        api_attn = WindowAttentionAPI(dim=dim, window_size=window_size, num_heads=num_heads)
        
        # Check relative position index
        impl_index = impl_attn.relative_position_index
        api_index = api_attn.relative_position_index
        
        print(f"Relative position index shape:")
        print(f"  - Implemented: {impl_index.shape}")
        print(f"  - API: {api_index.shape}")
        
        index_match = torch.equal(impl_index, api_index)
        print(f"  - Indices match: {index_match}")
        
        # Check bias table initialization
        print(f"\nRelative position bias table statistics:")
        print(f"  - Implemented: mean={impl_attn.relative_position_bias_table.mean().item():.6f}, "
              f"std={impl_attn.relative_position_bias_table.std().item():.6f}")
        print(f"  - API: mean={api_attn.relative_position_bias_table.mean().item():.6f}, "
              f"std={api_attn.relative_position_bias_table.std().item():.6f}")
        
        if index_match:
            print("\n✓ PASSED")
        else:
            print("\n✗ FAILED")
        
        return index_match
        
    except Exception as e:
        print(f"\n✗ FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_different_configurations(test_name=""):
    """Test various configurations"""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    configs_API = [
        {"dim": 96, "window_size": (7, 7), "num_heads": 3, "qkv_bias": True},
        {"dim": 192, "window_size": (7, 7), "num_heads": 6, "qkv_bias": True},
        {"dim": 96, "window_size": (8, 8), "num_heads": 3, "qkv_bias": False},
        {"dim": 384, "window_size": (7, 7), "num_heads": 12, "qkv_bias": True},
    ]

    configs_IMP = [
        {"dim": 96, "window_size": (7, 7), "num_heads": 3, "qkv_bias_enabled": True},
        {"dim": 192, "window_size": (7, 7), "num_heads": 6, "qkv_bias_enabled": True},
        {"dim": 96, "window_size": (8, 8), "num_heads": 3, "qkv_bias_enabled": False},
        {"dim": 384, "window_size": (7, 7), "num_heads": 12, "qkv_bias_enabled": True},
    ]

    all_passed = True

    try:
        from WindowAttention import WindowAttention as WindowAttentionImpl
        from original_swin import WindowAttention as WindowAttentionAPI

        for i, (config_IMP, config_API) in enumerate(zip(configs_IMP, configs_API)):
            print(f"\nConfiguration {i+1}: {config_IMP}")

            try:
                impl_attn = WindowAttentionImpl(**config_IMP)
                api_attn = WindowAttentionAPI(**config_API)

                    # Quick structure check
                match = (
                        impl_attn.dim == api_attn.dim and
                        impl_attn.num_heads == api_attn.num_heads
                )
                
                print(f"  ✓ Created successfully - Structure match: {match}")
                all_passed &= match
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                all_passed = False
        
        if all_passed:
            print("\n✓ PASSED - All configurations work")
        else:
            print("\n✗ FAILED - Some configurations failed")
        
        return all_passed
        
    except Exception as e:
        print(f"\n✗ FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run comprehensive test suite"""
    print("="*80)
    print("COMPARING Implemented WindowAttention vs API WindowAttention")
    print("="*80)
    
    all_passed = True
    
    all_passed &= test_initialization("Initialization and Structure")
    all_passed &= test_relative_position_bias("Relative Position Bias")
    all_passed &= test_flops_calculation("FLOPS Calculation")
    all_passed &= test_forward_pass_no_mask("Forward Pass (No Mask)")
    all_passed &= test_forward_pass_with_mask("Forward Pass (With Mask)")
    all_passed &= test_different_configurations("Different Configurations")
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return all_passed
# ==========================================================================================================


# ==========================================================================================================
# CRITICAL ISSUES IN YOUR IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
def print_critical_issues():
    """Print critical issues found in the implementation"""
    print("\n" + "="*80)
    print("CRITICAL ISSUES IN IMPLEMENTED VERSION")
    print("="*80)
    
    issues = [
        {
            "severity": "HIGH",
            "issue": "Device Mismatch",
            "description": "Converting to CPU numpy then back to device may cause issues",
            "location": "Lines: x_np = x.detach().cpu().numpy()",
            "fix": "Ensure .to(x.device) is called after torch.from_numpy()"
        },
        {
            "severity": "HIGH", 
            "issue": "Gradient Flow",
            "description": ".detach() breaks gradient flow - won't work for training",
            "location": "Lines: x.detach().cpu().numpy()",
            "fix": "Remove .detach() if you need gradients, or document inference-only"
        },
        {
            "severity": "MEDIUM",
            "issue": "Performance Overhead",
            "description": "Multiple torch->numpy->torch conversions add significant overhead",
            "location": "Forward pass converts 2x (qkv and proj)",
            "fix": "Keep everything in PyTorch or NumPy, not both"
        },
        {
            "severity": "MEDIUM",
            "issue": "Float Type Consistency",
            "description": ".float() call may cause precision issues",
            "location": "torch.from_numpy(qkv_np).to(x.device).float()",
            "fix": "Match original tensor dtype instead of forcing float()"
        },
        {
            "severity": "LOW",
            "issue": "Parameter Name Typo",
            "description": "Docstring has duplicate 'qkv_bias' parameter",
            "location": "Class docstring",
            "fix": "Second should be 'qk_scale'"
        }
    ]
    
    for issue in issues:
        print(f"\n[{issue['severity']}] {issue['issue']}")
        print(f"  Description: {issue['description']}")
        print(f"  Location: {issue['location']}")
        print(f"  Fix: {issue['fix']}")
    
    print("\n" + "="*80)


def print_recommendations():
    """Print recommendations for improvement"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. CHOOSE ONE FRAMEWORK:")
    print("   Option A: Pure PyTorch (recommended for training)")
    print("   Option B: Pure NumPy (for inference/understanding)")
    print("   Current: Mixing both causes overhead and complexity")
    
    print("\n2. IF KEEPING EXPLICITLINEAR:")
    print("   - Remove .detach() to allow gradients")
    print("   - Handle device placement consistently")
    print("   - Document that this is inference-only if using detach")
    
    print("\n3. PERFORMANCE:")
    print("   - Torch->NumPy conversion: ~10-100x slower")
    print("   - Consider implementing attention entirely in NumPy")
    print("   - Or use pure PyTorch without custom Linear")
    
    print("\n4. TESTING:")
    print("   - Test on GPU if available")
    print("   - Test gradient flow if needed for training")
    print("   - Compare inference speed")
    
    print("="*80)
# ==========================================================================================================


# ==========================================================================================================
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print_critical_issues()
    print_recommendations()
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    print("\nNote: Tests will fail if dependencies are not available.")
    print("Ensure: ExplicitLinear, Softmax, and original_swin are importable.")
    print("="*80)
    
    # Uncomment to run when dependencies are ready:
    run_all_tests()
    
    # print("\n" + "="*80)
    # print("SUMMARY")
    # print("="*80)
    # print("Your implementation uses ExplicitLinear and custom Softmax,")
    # print("but converts between PyTorch and NumPy in the forward pass.")
    # print("This works but has significant performance and design issues.")
    # print("\nFor production: Use pure PyTorch or pure NumPy, not mixed.")
    # print("="*80)