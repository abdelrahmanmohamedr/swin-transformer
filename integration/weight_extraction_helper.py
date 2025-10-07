"""
Weight Extraction Helper for Swin Transformer Comparison
Extracts weights from PyTorch model and prepares them for Manual implementation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List


"""
Weight Extraction Helper for Swin Transformer Comparison
Extracts weights from PyTorch model and prepares them for Manual implementation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List


def extract_linear_weights(linear_layer: nn.Linear) -> Dict:
    """Extract weights from nn.Linear layer"""
    weights = {
        'weight': linear_layer.weight.detach().cpu().numpy(),
    }
    if linear_layer.bias is not None:
        weights['bias'] = linear_layer.bias.detach().cpu().numpy()
    else:
        weights['bias'] = None
    return weights


def extract_layernorm_weights(norm_layer: nn.LayerNorm) -> Dict:
    """Extract weights from nn.LayerNorm"""
    return {
        'weight': norm_layer.weight.detach().cpu().numpy(),
        'bias': norm_layer.bias.detach().cpu().numpy()
    }


def extract_conv2d_weights(conv_layer: nn.Conv2d) -> Dict:
    """Extract weights from nn.Conv2d"""
    weights = {
        'weight': conv_layer.weight.detach().cpu().numpy(),
    }
    if conv_layer.bias is not None:
        weights['bias'] = conv_layer.bias.detach().cpu().numpy()
    else:
        weights['bias'] = None
    return weights


def extract_attention_weights(attn_module) -> Dict:
    """Extract all weights from WindowAttention module"""
    return {
        'qkv_weight': attn_module.qkv.weight.detach().cpu().numpy(),
        'qkv_bias': attn_module.qkv.bias.detach().cpu().numpy() if attn_module.qkv.bias is not None else None,
        'proj_weight': attn_module.proj.weight.detach().cpu().numpy(),
        'proj_bias': attn_module.proj.bias.detach().cpu().numpy() if attn_module.proj.bias is not None else None,
        'relative_position_bias_table': attn_module.relative_position_bias_table.detach().cpu().numpy(),
        'relative_position_index': attn_module.relative_position_index.detach().cpu().numpy()
    }


def extract_mlp_weights(mlp_module) -> Dict:
    """Extract weights from MLP module"""
    return {
        'fc1_weight': mlp_module.fc1.weight.detach().cpu().numpy(),
        'fc1_bias': mlp_module.fc1.bias.detach().cpu().numpy() if mlp_module.fc1.bias is not None else None,
        'fc2_weight': mlp_module.fc2.weight.detach().cpu().numpy(),
        'fc2_bias': mlp_module.fc2.bias.detach().cpu().numpy() if mlp_module.fc2.bias is not None else None,
    }


def extract_block_weights(block, block_idx: int) -> Dict:
    """Extract weights from a single SwinTransformerBlock"""
    weights = {
        'shift_size': block.shift_size,
        'norm1_weight': block.norm1.weight.detach().cpu().numpy(),
        'norm1_bias': block.norm1.bias.detach().cpu().numpy(),
        'attn_weights': extract_attention_weights(block.attn),
        'norm2_weight': block.norm2.weight.detach().cpu().numpy(),
        'norm2_bias': block.norm2.bias.detach().cpu().numpy(),
        'mlp_fc1_weight': block.mlp.fc1.weight.detach().cpu().numpy(),
        'mlp_fc1_bias': block.mlp.fc1.bias.detach().cpu().numpy() if block.mlp.fc1.bias is not None else None,
        'mlp_fc2_weight': block.mlp.fc2.weight.detach().cpu().numpy(),
        'mlp_fc2_bias': block.mlp.fc2.bias.detach().cpu().numpy() if block.mlp.fc2.bias is not None else None,
        'attn_mask': block.attn_mask.detach().cpu().numpy() if block.attn_mask is not None else None
    }
    return weights


def extract_layer_weights(layer, layer_idx: int) -> Dict:
    """Extract weights from a complete layer (BasicLayer)"""
    weights = {
        'blocks': []
    }

    # Extract weights from all blocks
    for block_idx, block in enumerate(layer.blocks):
        block_weights = extract_block_weights(block, block_idx)
        weights['blocks'].append(block_weights)

    # Extract downsample weights if exists
    if layer.downsample is not None:
        weights['downsample'] = {
            'reduction_weight': layer.downsample.reduction.weight.detach().cpu().numpy(),
            'norm_weight': layer.downsample.norm.weight.detach().cpu().numpy(),
            'norm_bias': layer.downsample.norm.bias.detach().cpu().numpy()
        }
    else:
        weights['downsample'] = None

    return weights


def extract_patch_embed_weights(patch_embed) -> Dict:
    """Extract weights from PatchEmbed module"""
    weights = {
        'proj_weight': patch_embed.proj.weight.detach().cpu().numpy(),
        'proj_bias': patch_embed.proj.bias.detach().cpu().numpy() if patch_embed.proj.bias is not None else None
    }

    if patch_embed.norm is not None:
        weights['norm_weight'] = patch_embed.norm.weight.detach().cpu().numpy()
        weights['norm_bias'] = patch_embed.norm.bias.detach().cpu().numpy()
    else:
        weights['norm_weight'] = None
        weights['norm_bias'] = None

    return weights


def extract_all_weights_from_pytorch(pytorch_model) -> Dict:
    """
    Extract all weights from PyTorch Swin Transformer model
    Returns a dictionary that can be used to initialize the Manual model
    """
    weights = {
        'patch_embed': extract_patch_embed_weights(pytorch_model.patch_embed),
        'layers': []
    }

    # Extract absolute position embedding if exists
    if pytorch_model.ape:
        weights['absolute_pos_embed'] = pytorch_model.absolute_pos_embed.detach().cpu().numpy()
    else:
        weights['absolute_pos_embed'] = None

    # Extract all layers
    for layer_idx, layer in enumerate(pytorch_model.layers):
        layer_weights = extract_layer_weights(layer, layer_idx)
        weights['layers'].append(layer_weights)

    # Extract final norm
    weights['norm'] = {
        'weight': pytorch_model.norm.weight.detach().cpu().numpy(),
        'bias': pytorch_model.norm.bias.detach().cpu().numpy()
    }

    # Extract classification head
    if not isinstance(pytorch_model.head, nn.Identity):
        weights['head'] = {
            'weight': pytorch_model.head.weight.detach().cpu().numpy(),
            'bias': pytorch_model.head.bias.detach().cpu().numpy() if pytorch_model.head.bias is not None else None
        }
    else:
        weights['head'] = None

    return weights


def create_small_swin_pytorch():
    """Create a small Swin Transformer for testing"""

    from original_swin import SwinTransformer

    model = SwinTransformer(
        img_size=56,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=48,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        fused_window_process=False
    )
    return model


def create_small_swin_manual(weights):
    """Create manual Swin Transformer with extracted weights"""

    from modified_swin_test import ManualSwinTransformer
    from LayerNorm_v4 import LayerNorm

    model = ManualSwinTransformer(
        img_size=56,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=48,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        fused_window_process=False,
        weights=weights
    )
    return model


def verify_weight_shapes(pytorch_model, extracted_weights):
    """Verify that extracted weights have correct shapes"""
    print("Verifying weight shapes...")
    issues = []

    # Check patch embed
    pt_conv = pytorch_model.patch_embed.proj.weight.shape
    ext_conv = extracted_weights['patch_embed']['proj_weight'].shape
    if pt_conv != ext_conv:
        issues.append(f"Patch embed conv: {pt_conv} vs {ext_conv}")

    # Check layers
    for layer_idx, (pt_layer, ext_layer) in enumerate(zip(pytorch_model.layers, extracted_weights['layers'])):
        # Check blocks
        for block_idx, (pt_block, ext_block) in enumerate(zip(pt_layer.blocks, ext_layer['blocks'])):
            # Check QKV
            pt_qkv = pt_block.attn.qkv.weight.shape
            ext_qkv = ext_block['attn_weights']['qkv_weight'].shape
            if pt_qkv != ext_qkv:
                issues.append(f"Layer {layer_idx} Block {block_idx} QKV: {pt_qkv} vs {ext_qkv}")

            # Check MLP FC1
            pt_fc1 = pt_block.mlp.fc1.weight.shape
            ext_fc1 = ext_block['mlp_fc1_weight'].shape
            if pt_fc1 != ext_fc1:
                issues.append(f"Layer {layer_idx} Block {block_idx} FC1: {pt_fc1} vs {ext_fc1}")

    if issues:
        print("Found shape mismatches:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("All weight shapes match! ✓")
        return True


def save_weights_to_file(weights, filename='swin_weights.npz'):
    """Save extracted weights to a file"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Weights saved to {filename}")


def load_weights_from_file(filename='swin_weights.npz'):
    """Load extracted weights from a file"""
    import pickle
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    print(f"Weights loaded from {filename}")
    return weights


def print_weight_statistics(weights):
    """Print statistics about extracted weights"""
    print("\n" + "="*80)
    print("Weight Statistics")
    print("="*80)

    def print_array_stats(name, arr):
        if arr is None:
            print(f"{name}: None")
        else:
            print(f"{name}:")
            print(f"  Shape: {arr.shape}")
            print(f"  Mean: {np.mean(arr):.6f}")
            print(f"  Std: {np.std(arr):.6f}")
            print(f"  Min: {np.min(arr):.6f}")
            print(f"  Max: {np.max(arr):.6f}")

    # Patch embed
    print("\nPatch Embed:")
    print_array_stats("  Conv Weight", weights['patch_embed']['proj_weight'])
    print_array_stats("  Conv Bias", weights['patch_embed']['proj_bias'])

    # Layers
    for layer_idx, layer in enumerate(weights['layers']):
        print(f"\nLayer {layer_idx}:")
        print(f"  Number of blocks: {len(layer['blocks'])}")

        for block_idx, block in enumerate(layer['blocks']):
            print(f"\n  Block {block_idx}:")
            print_array_stats("    QKV Weight", block['attn_weights']['qkv_weight'])
            print_array_stats("    MLP FC1 Weight", block['mlp_fc1_weight'])
            print_array_stats("    MLP FC2 Weight", block['mlp_fc2_weight'])

    # Final norm
    print("\nFinal Norm:")
    print_array_stats("  Weight", weights['norm']['weight'])
    print_array_stats("  Bias", weights['norm']['bias'])

    # Head
    if weights['head'] is not None:
        print("\nClassification Head:")
        print_array_stats("  Weight", weights['head']['weight'])
        print_array_stats("  Bias", weights['head']['bias'])


def compare_model_outputs_simple(pytorch_model, manual_model, num_samples=5):
    """Simple comparison of model outputs"""
    print("\n" + "="*80)
    print("Quick Output Comparison")
    print("="*80)

    pytorch_model.eval()

    for i in range(num_samples):
        # Create random input
        x = torch.randn(1, 3, 56, 56)

        # Get outputs
        with torch.no_grad():
            pt_out = pytorch_model(x)
        manual_out = manual_model(x)

        # Convert to numpy
        if isinstance(manual_out, torch.Tensor):
            manual_out = manual_out.detach().cpu().numpy()
        pt_out_np = pt_out.detach().cpu().numpy()

        # Compare
        max_diff = np.max(np.abs(pt_out_np - manual_out))
        mean_diff = np.mean(np.abs(pt_out_np - manual_out))

        pt_pred = np.argmax(pt_out_np)
        manual_pred = np.argmax(manual_out)

        match = "✓" if pt_pred == manual_pred else "✗"

        print(f"Sample {i+1}:")
        print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"  PyTorch pred: {pt_pred}, Manual pred: {manual_pred} {match}")


if __name__ == "__main__":
    print("="*80)
    print("Weight Extraction and Model Setup")
    print("="*80)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create PyTorch model
    print("\n1. Creating PyTorch model...")
    pytorch_model = create_small_swin_pytorch()
    pytorch_model.eval()

    total_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Extract weights
    print("\n2. Extracting weights...")
    weights = extract_all_weights_from_pytorch(pytorch_model)
    print(f"   Extracted weights for {len(weights['layers'])} layers")

    # Verify shapes
    print("\n3. Verifying weight shapes...")
    verify_weight_shapes(pytorch_model, weights)

    # Print statistics
    print_weight_statistics(weights)

    # Create manual model
    print("\n4. Creating Manual model with extracted weights...")
    manual_model = create_small_swin_manual(weights)

    # Quick comparison
    print("\n5. Quick output comparison...")
    compare_model_outputs_simple(pytorch_model, manual_model, num_samples=3)

    print("\n" + "="*80)
    print("Setup complete! You can now run the full test suite.")
    print("="*80)