import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from conv2d import MyConv2d as ManualConv2d
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ============================================================================
# Original PatchEmbed Implementation
# ============================================================================

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
# ============================================================================
# Manaul PatchEmbed Implementation
# ============================================================================

class PatchEmbedManual(nn.Module):
    r""" Image to Patch Embedding with Manual Conv2d

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        weight (np.ndarray, optional): Pre-extracted conv weights
        bias (np.ndarray, optional): Pre-extracted conv bias
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, 
                 norm_layer=None, weight=None, bias=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Create ManualConv2d with the provided weights
        # ManualConv2d handles weight initialization internally
        self.proj = ManualConv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            weight=weight, 
            bias=bias,
            bias_condition=True  # PatchEmbed always has bias
        )
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # ManualConv2d returns a PyTorch tensor
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def compare_patch_embed():
    print("="*80)
    print("Comparing PatchEmbed: nn.Conv2d vs ManualConv2d")
    print("="*80 + "\n")
    
    torch.manual_seed(42)
    
    # Configuration
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 768
    batch_size = 4
    
    print(f"Configuration:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Input channels: {in_chans}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Expected patches: {(img_size // patch_size) ** 2}")
    print()
    
    # Create models
    patch_embed_original = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=None
    )
    
    patch_embed_manual = PatchEmbedManual(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=None
    )
    
    # Copy weights from original to manual to ensure same parameters
    print("Copying weights from nn.Conv2d to Manual..")
    patch_embed_manual.proj.weight.data = patch_embed_original.proj.weight.data.clone()
    patch_embed_manual.proj.bias.data = patch_embed_original.proj.bias.data.clone()
    print("✓ Weights copied\n")
    
    # Create random input
    x = torch.randn(batch_size, in_chans, img_size, img_size)
    
    print("-"*80)
    print("Running forward pass...")
    print("-"*80)
    
    # Forward pass
    with torch.no_grad():
        output_original = patch_embed_original(x)
        output_manual = patch_embed_manual(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  [batch_size, channels, height, width]")
    print(f"\nOriginal PatchEmbed output shape: {output_original.shape}")
    print(f"Manual PatchEmbed output shape: {output_manual.shape}")
    print(f"  [batch_size, num_patches, embed_dim]")
    
    # Compare outputs
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Check shapes
    shape_match = output_original.shape == output_manual.shape
    print(f"\n✓ Shape Match: {shape_match}")
    
    # Check values
    max_diff = torch.max(torch.abs(output_original - output_manual)).item()
    mean_diff = torch.mean(torch.abs(output_original - output_manual)).item()
    
    print(f"\nNumerical Comparison:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    # Tolerance check
    tolerance = 1e-5
    values_match = torch.allclose(output_original, output_manual, atol=tolerance)
    print(f"  Values match (atol={tolerance}): {values_match}")
    
    # Sample values comparison
    print(f"\nSample Output Comparison (first 5 values of first patch, first batch):")
    print(f"  Original: {output_original[0, 0, :5].numpy()}")
    print(f"  Manual:   {output_manual[0, 0, :5].numpy()}")
    print(f"  Diff:     {(output_original[0, 0, :5] - output_manual[0, 0, :5]).numpy()}")
    
    # Test with different configurations
    print("\n" + "="*80)
    print("Testing Different Configurations")
    print("="*80)
    
    configs = [
        {"img_size": 224, "patch_size": 16, "in_chans": 3, "embed_dim": 768},
        {"img_size": 224, "patch_size": 32, "in_chans": 3, "embed_dim": 512},
        {"img_size": 96, "patch_size": 4, "in_chans": 3, "embed_dim": 96},
        {"img_size": 64, "patch_size": 8, "in_chans": 1, "embed_dim": 128},
    ]
    
    all_passed = True
    for i, config in enumerate(configs, 1):
        print(f"\nTest {i}: img_size={config['img_size']}, patch_size={config['patch_size']}, "
              f"in_chans={config['in_chans']}, embed_dim={config['embed_dim']}")
        
        pe_orig = PatchEmbed(**config)
        pe_manual = PatchEmbedManual(**config)
        
        # Copy weights
        pe_manual.proj.weight.data = pe_orig.proj.weight.data.clone()
        pe_manual.proj.bias.data = pe_orig.proj.bias.data.clone()
        
        # Test
        x_test = torch.randn(2, config['in_chans'], config['img_size'], config['img_size'])
        
        with torch.no_grad():
            out_orig = pe_orig(x_test)
            out_manual = pe_manual(x_test)
        
        max_diff = torch.max(torch.abs(out_orig - out_manual)).item()
        match = torch.allclose(out_orig, out_manual, atol=1e-5)
        
        print(f"  Output shape: {out_orig.shape}")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Match: {'✓' if match else '✗'}")
        
        if not match:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 SUCCESS! All tests passed!")
        print("ManualConv2d produces identical results to nn.Conv2d in PatchEmbed")
    else:
        print("❌ Some tests failed!")
    print("="*80)


if __name__ == "__main__":
    compare_patch_embed()