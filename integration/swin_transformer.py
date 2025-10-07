
import numpy as np
import time
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from BasicLayer import ManualBasicLayer
from PatchMerging import ManualPatchMerging
from patch_embed import PatchEmbedManual
from adaptive_avg_pool import AdaptiveAvgPool1d
from flatten import custom_flatten
from constant_ import constant_
from LayerNorm_v4 import LayerNorm
from linspace import linspace_list
from ModuleList import ModuleList
from linear import ExplicitLinear
from typing import Dict, Tuple
from original_swin import SwinTransformer as SwinTransformerAPI
from SwinTransformerBlock import ManualSwinTransformerBlock

class ManualSwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
        weights (dict, optional): Dictionary containing pre-extracted weights from PyTorch model
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 weights=None, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Extract weights if provided
        patch_embed_weights = None
        absolute_pos_embed_weights = None
        layer_weights_list = None
        norm_weights = None
        head_weights = None

        if weights is not None:
            patch_embed_weights = weights.get('patch_embed')
            absolute_pos_embed_weights = weights.get('absolute_pos_embed')
            layer_weights_list = weights.get('layers', [])
            norm_weights = weights.get('norm')
            head_weights = weights.get('head')

        # Split image into non-overlapping patches
        proj_weight = None
        proj_bias = None

        if patch_embed_weights is not None:
            proj_weight = patch_embed_weights.get('proj_weight')
            proj_bias = patch_embed_weights.get('proj_bias')

        self.patch_embed = PatchEmbedManual(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=LayerNorm if self.patch_norm else None,
            weight=proj_weight,
            bias=proj_bias
        )

        # If patch_norm is True and we have norm weights, set them
        if self.patch_norm and patch_embed_weights is not None:
            if 'norm_weight' in patch_embed_weights and patch_embed_weights['norm_weight'] is not None:
                self.patch_embed.norm.weight = patch_embed_weights['norm_weight']
            if 'norm_bias' in patch_embed_weights and patch_embed_weights['norm_bias'] is not None:
                self.patch_embed.norm.bias = patch_embed_weights['norm_bias']


        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            if absolute_pos_embed_weights is not None:
                with torch.no_grad():
                    self.absolute_pos_embed.copy_(torch.from_numpy(absolute_pos_embed_weights))
            else:
                trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.tensor(linspace_list(0, drop_path_rate, sum(depths)))]

        # Build layers
        self.layers = ModuleList()
        for i_layer in range(self.num_layers):
            # Get weights for this layer if available
            layer_weights = None
            if layer_weights_list is not None and i_layer < len(layer_weights_list):
                layer_weights = layer_weights_list[i_layer]

            # Extract blocks and downsample weights
            blocks_weights = None
            downsample_weights = None
            if layer_weights is not None:
                blocks_weights = layer_weights.get('blocks')
                downsample_weights = layer_weights.get('downsample')

            layer = ManualBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                 patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=ManualPatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
                blocks_weights=blocks_weights,
                downsample_weights=downsample_weights
            )
            self.layers.append(layer)

        # Final norm
        self.norm = LayerNorm(normalized_shape=self.num_features)
        if norm_weights is not None:
            # For NumPy-based LayerNorm, directly assign NumPy arrays
            if 'weight' in norm_weights:
                self.norm.weight = norm_weights['weight']
            if 'bias' in norm_weights:
                self.norm.bias = norm_weights['bias']


        # Average pooling and classification head
        self.avgpool = AdaptiveAvgPool1d(1)
        self.head = ExplicitLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if head_weights is not None and num_classes > 0:
            with torch.no_grad():
                if 'weight' in head_weights:
                    weight_tensor = torch.from_numpy(head_weights['weight'])
                    if isinstance(self.head.weight, nn.Parameter):
                        self.head.weight.data.copy_(weight_tensor)
                    else:
                         # If it's a NumPy array, replace it
                        self.head.weight = head_weights['weight']

                if 'bias' in head_weights and head_weights['bias'] is not None:
                    bias_tensor = torch.from_numpy(head_weights['bias'])
                    if isinstance(self.head.bias, nn.Parameter):
                        self.head.bias.data.copy_(bias_tensor)
                    else:
                         # If it's a NumPy array, replace it
                        self.head.bias = head_weights['bias']


        # Initialize weights if not provided
        if weights is None:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, ExplicitLinear):
            # ExplicitLinear handles its own weight initialization if not provided
            pass
        elif isinstance(m, LayerNorm):
            # LayerNorm handles its own weight initialization if not provided
            pass
        elif isinstance(m, PatchEmbedManual):
            # PatchEmbedManual handles its own weight initialization if not provided
            pass
        elif isinstance(m, ManualSwinTransformerBlock):
             # ManualSwinTransformerBlock handles its own weight initialization if not provided
             pass
        elif isinstance(m, ManualBasicLayer):
             # ManualBasicLayer handles its own weight initialization if not provided
             pass
        elif isinstance(m, ManualPatchMerging):
             # ManualPatchMerging handles its own weight initialization if not provided
             pass
        elif isinstance(m, nn.Parameter):
             # Handle potential remaining nn.Parameters
             if m.numel() > 1: # Avoid single element parameters which might be biases handled elsewhere
                 trunc_normal_(m, std=.02)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        print(f"--- forward_features start, input type: {type(x)}, shape: {x.shape}")
        # Ensure x is a tensor for consistency at the input
        if not isinstance(x, torch.Tensor):
            print("Converting input to torch tensor")
            x = torch.from_numpy(x).float()
        print(f"After input conversion, type: {type(x)}, shape: {x.shape}")

        # Patch embed returns a tensor
        print("Calling patch_embed...")
        x = self.patch_embed(x)
        print(f"After patch_embed, type: {type(x)}, shape: {x.shape}")


        # Ensure absolute_pos_embed is a tensor and on the same device/dtype
        if self.ape:
            print("Applying APE...")
            if not isinstance(self.absolute_pos_embed, torch.Tensor):
                 print("Converting APE to torch tensor")
                 self.absolute_pos_embed = torch.from_numpy(self.absolute_pos_embed).to(x.device, x.dtype)
            x = x + self.absolute_pos_embed
            print(f"After APE, type: {type(x)}, shape: {x.shape}")


        x = self.pos_drop(x)
        print(f"After pos_drop, type: {type(x)}, shape: {x.shape}")


        for i, layer in enumerate(self.layers):
            print(f"Calling layer {i}...")
            # ManualBasicLayer should handle tensor input and output tensor
            x = layer(x)
            print(f"After layer {i}, type: {type(x)}, shape: {x.shape}")


        # The norm should handle tensors properly with your implementation
        # Assuming LayerNorm handles both numpy and torch
        print("Calling final norm...")
        x = self.norm(x)  # B L C
        print(f"After final norm, type: {type(x)}, shape: {x.shape}")


        # Ensure we're working with tensors for the remaining operations
        if not isinstance(x, torch.Tensor):
            print("Converting final norm output to torch tensor")
            x = torch.from_numpy(x).float()
        print(f"Before transpose, type: {type(x)}, shape: {x.shape}")


        x = x.transpose(1, 2)  # B C L
        print(f"After transpose, type: {type(x)}, shape: {x.shape}")

        x = self.avgpool(x)  # B C 1
        print(f"After avgpool, type: {type(x)}, shape: {x.shape}")

        x = torch.flatten(x, 1)  # B C
        print(f"After flatten, type: {type(x)}, shape: {x.shape}")
        print("--- forward_features end ---")
        return x

    def forward(self, x):
        print(f"--- Full forward start, input type: {type(x)}, shape: {x.shape}")
        x = self.forward_features(x)
        print(f"After forward_features, type: {type(x)}, shape: {x.shape}")


        # Handle classification head which expects NumPy input
        if not isinstance(self.head, nn.Identity):
            print("Calling classification head...")
            # Ensure x is a NumPy array for ExplicitLinear
            if isinstance(x, torch.Tensor):
                print("Converting forward_features output to numpy for head")
                device = x.device
                dtype = x.dtype
                x_np = x.detach().cpu().numpy()
                x_out_np = self.head(x_np)
                # Convert back to tensor to match PyTorch implementation
                print("Converting head output back to torch tensor")
                x = torch.from_numpy(x_out_np).to(device=device, dtype=dtype)
            else:
                print("Input to head is already numpy")
                # If input was already numpy, head returns numpy
                x = self.head(x)
                # Convert to tensor before returning to match PyTorch output type
                print("Converting head output to torch tensor")
                x = torch.from_numpy(x).float()
        else:
            print("Head is nn.Identity, passing through")
            x = self.head(x) # Identity returns the input as is (tensor)

        print(f"--- Full forward end, output type: {type(x)}, shape: {x.shape}")
        return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        # Flops for final norm, avgpool, and head
        # The provided flops calculation for SwinTransformer seems to have a slight discrepancy
        # Let's replicate the original calculation's structure
        Ho, Wo = self.patches_resolution
        num_patches = Ho * Wo
        # Flops for final norm
        flops += self.num_features * num_patches // (2 ** (self.num_layers -1)) # This part seems off in original
        # Replicating original structure:
        final_resolution = (self.patches_resolution[0] // (2 ** (self.num_layers -1)),
                            self.patches_resolution[1] // (2 ** (self.num_layers -1)))
        final_num_patches = final_resolution[0] * final_resolution[1]
        flops += self.num_features * final_num_patches # Norm after the last layer

        # Avgpool: B x C x 1, effectively a reduction along L dimension (size final_num_patches)
        # Flops are typically considered negligible or zero for pooling
        # If we consider it as sum/count: B * C * final_num_patches

        # Head: B x C -> B x num_classes
        flops += self.num_features * self.num_classes

        return flops


class SwinTransformerTester:
    """
    Comprehensive test suite for comparing Manual and PyTorch SwinTransformer implementations
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.0):
        """
        Initialize tester with configuration parameters
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # For reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def extract_all_weights(self, pytorch_model) -> Dict:
        """
        Extract all weights from PyTorch SwinTransformer

        Args:
            pytorch_model: PyTorch SwinTransformer instance

        Returns:
            Dictionary containing all extracted weights
        """
        weights = {
            'patch_embed': {},
            'absolute_pos_embed': None,
            'layers': [],
            'norm': {},
            'head': {}
        }

        # Patch embedding weights
        weights['patch_embed'] = {
            'proj_weight': pytorch_model.patch_embed.proj.weight.detach().cpu().numpy(),
            'proj_bias': pytorch_model.patch_embed.proj.bias.detach().cpu().numpy() if pytorch_model.patch_embed.proj.bias is not None else None,
            'norm_weight': pytorch_model.patch_embed.norm.weight.detach().cpu().numpy() if pytorch_model.patch_embed.norm is not None else None,
            'norm_bias': pytorch_model.patch_embed.norm.bias.detach().cpu().numpy() if pytorch_model.patch_embed.norm is not None else None,
        }

        # Absolute position embedding
        if pytorch_model.ape:
            weights['absolute_pos_embed'] = pytorch_model.absolute_pos_embed.detach().cpu().numpy()

        # Extract weights from each layer
        for layer_idx, layer in enumerate(pytorch_model.layers):
            layer_weights = {
                'blocks': [],
                'downsample': None
            }

            # Extract blocks
            for block_idx, block in enumerate(layer.blocks):
                block_weights = {
                    'shift_size': block.shift_size,
                    'norm1_weight': block.norm1.weight.detach().cpu().numpy(),
                    'norm1_bias': block.norm1.bias.detach().cpu().numpy(),
                    'attn_weights': {
                        'qkv_weight': block.attn.qkv.weight.detach().cpu().numpy(),
                        'qkv_bias': block.attn.qkv.bias.detach().cpu().numpy() if block.attn.qkv.bias is not None else None,
                        'proj_weight': block.attn.proj.weight.detach().cpu().numpy(),
                        'proj_bias': block.attn.proj.bias.detach().cpu().numpy() if block.attn.proj.bias is not None else None,
                        'relative_position_bias_table': block.attn.relative_position_bias_table.detach().cpu().numpy()
                    },
                    'norm2_weight': block.norm2.weight.detach().cpu().numpy(),
                    'norm2_bias': block.norm2.bias.detach().cpu().numpy(),
                    'mlp_fc1_weight': block.mlp.fc1.weight.detach().cpu().numpy(),
                    'mlp_fc1_bias': block.mlp.fc1.bias.detach().cpu().numpy(),
                    'mlp_fc2_weight': block.mlp.fc2.weight.detach().cpu().numpy(),
                    'mlp_fc2_bias': block.mlp.fc2.bias.detach().cpu().numpy(),
                    'attn_mask': block.attn_mask.detach().cpu().numpy() if block.attn_mask is not None else None
                }
                layer_weights['blocks'].append(block_weights)

            # Extract downsample weights if present
            if layer.downsample is not None:
                layer_weights['downsample'] = {
                    'reduction_weight': layer.downsample.reduction.weight.detach().cpu().numpy(),
                    'norm_weight': layer.downsample.norm.weight.detach().cpu().numpy(),
                    'norm_bias': layer.downsample.norm.bias.detach().cpu().numpy()
                }

            weights['layers'].append(layer_weights)

        # Final norm
        weights['norm'] = {
            'weight': pytorch_model.norm.weight.detach().cpu().numpy(),
            'bias': pytorch_model.norm.bias.detach().cpu().numpy()
        }

        # Head
        if hasattr(pytorch_model.head, 'weight'):
            weights['head'] = {
                'weight': pytorch_model.head.weight.detach().cpu().numpy(),
                'bias': pytorch_model.head.bias.detach().cpu().numpy() if pytorch_model.head.bias is not None else None
            }

        return weights

    def create_test_input(self, batch_size=2) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Create test input data

        Args:
            batch_size: Batch size for test input

        Returns:
            Tuple of (PyTorch tensor, NumPy array) with same values
        """
        # Create random input image: B, C, H, W
        input_np = np.random.randn(batch_size, self.in_chans, self.img_size, self.img_size).astype(np.float32)
        input_torch = torch.from_numpy(input_np)

        return input_torch, input_np

    def test_forward_features(self, pytorch_model, manual_model, batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test forward_features (everything before classification head)
        """
        results = {
            'passed': False,
            'error': None
        }

        try:
            input_torch, input_np = self.create_test_input(batch_size)

            # PyTorch
            pytorch_model.eval()
            with torch.no_grad():
                start_time = time.time()
                pytorch_features = pytorch_model.forward_features(input_torch)
                results['pytorch_time'] = time.time() - start_time
            pytorch_features_np = pytorch_features.detach().cpu().numpy()

            # Manual
            start_time = time.time()
            # Pass the torch tensor to manual model, it should handle the conversion internally
            manual_features = manual_model.forward_features(input_torch)
            results['manual_time'] = time.time() - start_time

            # Ensure manual_features is numpy for comparison
            if isinstance(manual_features, torch.Tensor):
                manual_features_np = manual_features.detach().cpu().numpy()
            else:
                manual_features_np = manual_features

            # Compare
            diff = np.abs(pytorch_features_np - manual_features_np)
            results['max_diff'] = np.max(diff)
            results['mean_diff'] = np.mean(diff)
            results['median_diff'] = np.median(diff)
            results['pytorch_shape'] = pytorch_features_np.shape
            results['manual_shape'] = manual_features_np.shape
            results['passed'] = results['max_diff'] < tolerance

        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()

        return results

    def test_full_forward(self, pytorch_model, manual_model, batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test full forward pass (including classification head)
        """
        results = {
            'passed': False,
            'error': None
        }

        try:
            input_torch, input_np = self.create_test_input(batch_size)

            # PyTorch
            pytorch_model.eval()
            with torch.no_grad():
                start_time = time.time()
                pytorch_output = pytorch_model(input_torch)
                results['pytorch_time'] = time.time() - start_time
            pytorch_output_np = pytorch_output.detach().cpu().numpy()

            # Manual
            start_time = time.time()
            # Pass the torch tensor to manual model, it should handle the conversion internally
            manual_output = manual_model(input_torch)
            results['manual_time'] = time.time() - start_time

            # Ensure manual_output is numpy for comparison
            if isinstance(manual_output, torch.Tensor):
                manual_output_np = manual_output.detach().cpu().numpy()
            else:
                manual_output_np = manual_output


            # Compare
            diff = np.abs(pytorch_output_np - manual_output_np)
            results['max_diff'] = np.max(diff)
            results['mean_diff'] = np.mean(diff)
            results['median_diff'] = np.median(diff)
            results['pytorch_shape'] = pytorch_output_np.shape
            results['manual_shape'] = manual_output_np.shape
            results['passed'] = results['max_diff'] < tolerance

        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()

        return results

    def test_multiple_batch_sizes(self, pytorch_model, manual_model,
                                  batch_sizes=[1, 2, 4], tolerance=1e-4) -> Dict:
        """
        Test with different batch sizes
        """
        results = {}

        for batch_size in batch_sizes:
            try:
                input_torch, input_np = self.create_test_input(batch_size)

                # PyTorch
                pytorch_model.eval()
                with torch.no_grad():
                    output_torch = pytorch_model(input_torch)
                output_torch_np = output_torch.detach().cpu().numpy()

                # Manual
                output_manual = manual_model(input_torch) # Pass torch tensor
                if isinstance(output_manual, torch.Tensor):
                    output_manual_np = output_manual.detach().cpu().numpy()
                else:
                    output_manual_np = output_manual

                diff = np.abs(output_torch_np - output_manual_np)

                results[f'batch_{batch_size}'] = {
                    'pytorch_shape': tuple(output_torch_np.shape),
                    'manual_shape': tuple(output_manual_np.shape),
                    'max_diff': np.max(diff),
                    'mean_diff': np.mean(diff),
                    'passed': np.max(diff) < tolerance,
                    'error': None
                }
            except Exception as e:
                results[f'batch_{batch_size}'] = {
                    'error': str(e)
                }

        return results
    
    def test_patch_embed(self, pytorch_model, manual_model, batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test patch embedding layer
        """
        results = {
            'passed': False,
            'error': None
        }

        try:
            input_torch, input_np = self.create_test_input(batch_size)

            print(f"\n  Testing with input shape: {input_torch.shape}")
            print(f"  Input type: {type(input_torch)}")

            # PyTorch
            pytorch_model.eval()
            with torch.no_grad():
                print("  Running PyTorch patch_embed...")
                pytorch_embed = pytorch_model.patch_embed(input_torch)
                print(f"  PyTorch output shape: {pytorch_embed.shape}, type: {type(pytorch_embed)}")
            pytorch_embed_np = pytorch_embed.detach().cpu().numpy()

            # Manual
            print("  Running Manual patch_embed...")
            manual_embed = manual_model.patch_embed(input_torch if isinstance(manual_model.patch_embed, nn.Module) else input_np)
            print(f"  Manual output type: {type(manual_embed)}")

            if isinstance(manual_embed, torch.Tensor):
                print(f"  Manual output is tensor, shape: {manual_embed.shape}")
                manual_embed_np = manual_embed.detach().cpu().numpy()
            else:
                print(f"  Manual output is not tensor: {type(manual_embed)}")
                manual_embed_np = manual_embed

            # Compare
            diff = np.abs(pytorch_embed_np - manual_embed_np)
            results['max_diff'] = np.max(diff)
            results['mean_diff'] = np.mean(diff)
            results['pytorch_shape'] = pytorch_embed_np.shape
            results['manual_shape'] = manual_embed_np.shape
            results['passed'] = results['max_diff'] < tolerance

        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
            print(f"\n  DETAILED ERROR:")
            print(results['traceback'])

        return results
    
    def test_per_layer(self, pytorch_model, manual_model, batch_size=2, tolerance=1e-4) -> Dict:
        """
        Test each layer individually
        """
        results = {}
        
        try:
            input_torch, input_np = self.create_test_input(batch_size)
            
            print(f"\n  Testing per-layer with input shape: {input_torch.shape}")
            
            # Get embeddings
            pytorch_model.eval()
            with torch.no_grad():
                x_torch = pytorch_model.patch_embed(input_torch)
                if pytorch_model.ape:
                    x_torch = x_torch + pytorch_model.absolute_pos_embed
                x_torch = pytorch_model.pos_drop(x_torch)
                print(f"  After patch_embed: {x_torch.shape}, type: {type(x_torch)}")
            
            x_manual = manual_model.patch_embed(input_torch if isinstance(manual_model.patch_embed, nn.Module) else input_np)
            print(f"  Manual after patch_embed: type: {type(x_manual)}")
            
            if isinstance(x_manual, torch.Tensor):
                x_manual_np = x_manual.detach().cpu().numpy()
            else:
                x_manual_np = x_manual
            
            # Test each layer
            for i, (pytorch_layer, manual_layer) in enumerate(zip(pytorch_model.layers, manual_model.layers)):
                layer_result = {}
                
                try:
                    print(f"\n  Testing layer {i}...")
                    print(f"    Input to layer {i}: type={type(x_torch)}, shape={x_torch.shape}")
                    
                    # PyTorch layer
                    with torch.no_grad():
                        out_torch = pytorch_layer(x_torch)
                    print(f"    PyTorch layer {i} output: type={type(out_torch)}, shape={out_torch.shape}")
                    out_torch_np = out_torch.detach().cpu().numpy()
                    
                    # Manual layer
                    print(f"    Running manual layer {i}...")
                    if isinstance(manual_layer, nn.Module):
                        out_manual = manual_layer(x_torch)
                    else:
                        out_manual = manual_layer.forward(x_manual_np)
                    
                    print(f"    Manual layer {i} output: type={type(out_manual)}")
                    
                    if isinstance(out_manual, torch.Tensor):
                        print(f"    Manual output is tensor, shape: {out_manual.shape}")
                        out_manual_np = out_manual.detach().cpu().numpy()
                    else:
                        print(f"    Manual output is not tensor: {type(out_manual)}")
                        out_manual_np = out_manual
                    
                    # Compare
                    diff = np.abs(out_torch_np - out_manual_np)
                    layer_result['max_diff'] = np.max(diff)
                    layer_result['mean_diff'] = np.mean(diff)
                    layer_result['pytorch_shape'] = out_torch_np.shape
                    layer_result['manual_shape'] = out_manual_np.shape
                    layer_result['passed'] = layer_result['max_diff'] < tolerance
                    layer_result['error'] = None
                    
                    # Update for next layer
                    x_torch = out_torch
                    x_manual_np = out_manual_np
                    
                except Exception as e:
                    layer_result['error'] = str(e)
                    import traceback
                    layer_result['traceback'] = traceback.format_exc()
                    print(f"\n    DETAILED ERROR in layer {i}:")
                    print(layer_result['traceback'])
                
                results[f'layer_{i}'] = layer_result
                
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
            print(f"\n  DETAILED ERROR in per-layer test:")
            print(results['traceback'])
        
        return results

    def run_all_tests(self, pytorch_model, manual_model, verbose=True) -> Dict:
        """
        Run comprehensive test suite
        """
        if verbose:
            print("=" * 80)
            print("SWIN TRANSFORMER TEST SUITE")
            print("=" * 80)
            print(f"Configuration:")
            print(f"  img_size: {self.img_size}")
            print(f"  patch_size: {self.patch_size}")
            print(f"  in_chans: {self.in_chans}")
            print(f"  num_classes: {self.num_classes}")
            print(f"  embed_dim: {self.embed_dim}")
            print(f"  depths: {self.depths}")
            print(f"  num_heads: {self.num_heads}")
            print(f"  window_size: {self.window_size}")
            print(f"  mlp_ratio: {self.mlp_ratio}")
            print("=" * 80)

        all_results = {}

        # Test 1: Patch Embedding
        if verbose:
            print("\n[Test 1] Patch Embedding")
        patch_results = self.test_patch_embed(pytorch_model, manual_model)
        all_results['patch_embed'] = patch_results

        if verbose:
            if patch_results['error']:
                print(f"  ERROR: {patch_results['error']}")
            else:
                print(f"  PyTorch shape: {patch_results['pytorch_shape']}")
                print(f"  Manual shape:  {patch_results['manual_shape']}")
                print(f"  Max diff:      {patch_results['max_diff']:.2e}")
                print(f"  Mean diff:     {patch_results['mean_diff']:.2e}")
                status = "PASS" if patch_results['passed'] else "FAIL"
                print(f"  Status:        {status}")

        # Test 2: Per-Layer Testing
        if verbose:
            print("\n[Test 2] Per-Layer Testing")
        layer_results = self.test_per_layer(pytorch_model, manual_model)
        all_results['per_layer'] = layer_results

        if verbose:
            if 'error' in layer_results:
                print(f"  ERROR: {layer_results['error']}")
            else:
                for layer_name, layer_result in layer_results.items():
                    if 'error' in layer_result and layer_result['error']:
                        print(f"  {layer_name}: ERROR - {layer_result['error']}")
                    else:
                        status = "PASS" if layer_result['passed'] else "FAIL"
                        print(f"  {status}: {layer_name} - max_diff={layer_result['max_diff']:.2e}")

        # Test 3: Forward Features
        if verbose:
            print("\n[Test 3] Forward Features (Before Classification Head)")
        features_results = self.test_forward_features(pytorch_model, manual_model)
        all_results['forward_features'] = features_results

        if verbose:
            if features_results['error']:
                print(f"  ERROR: {features_results['error']}")
            else:
                print(f"  PyTorch shape: {features_results['pytorch_shape']}")
                print(f"  Manual shape:  {features_results['manual_shape']}")
                print(f"  Max diff:      {features_results['max_diff']:.2e}")
                print(f"  Mean diff:     {features_results['mean_diff']:.2e}")
                print(f"  PyTorch time:  {features_results['pytorch_time']:.4f}s")
                print(f"  Manual time:   {features_results['manual_time']:.4f}s")
                status = "PASS" if features_results['passed'] else "FAIL"
                print(f"  Status:        {status}")

        # Test 4: Full Forward Pass
        if verbose:
            print("\n[Test 4] Full Forward Pass (With Classification Head)")
        forward_results = self.test_full_forward(pytorch_model, manual_model)
        all_results['full_forward'] = forward_results

        if verbose:
            if forward_results['error']:
                print(f"  ERROR: {forward_results['error']}")
            else:
                print(f"  PyTorch shape: {forward_results['pytorch_shape']}")
                print(f"  Manual shape:  {forward_results['manual_shape']}")
                print(f"  Max diff:      {forward_results['max_diff']:.2e}")
                print(f"  Mean diff:     {forward_results['mean_diff']:.2e}")
                print(f"  PyTorch time:  {forward_results['pytorch_time']:.4f}s")
                print(f"  Manual time:   {forward_results['manual_time']:.4f}s")
                status = "PASS" if forward_results['passed'] else "FAIL"
                print(f"  Status:        {status}")

        # Test 5: Multiple Batch Sizes
        if verbose:
            print("\n[Test 5] Multiple Batch Sizes")
        batch_results = self.test_multiple_batch_sizes(pytorch_model, manual_model)
        all_results['batch_sizes'] = batch_results

        if verbose:
            for batch_name, batch_result in batch_results.items():
                if 'error' in batch_result and batch_result['error']:
                    print(f"  {batch_name}: ERROR - {batch_result['error']}")
                else:
                    status = "PASS" if batch_result['passed'] else "FAIL"
                    print(f"  {status}: {batch_name} - max_diff={batch_result['max_diff']:.2e}")

        # Summary
        if verbose:
            print("\n" + "=" * 80)
            print("TEST SUMMARY")
            print("=" * 80)

            tests_passed = 0
            total_tests = 0

            # Count passed tests
            if not patch_results.get('error'):
                total_tests += 1
                if patch_results['passed']:
                    tests_passed += 1

            if not features_results.get('error'):
                total_tests += 1
                if features_results['passed']:
                    tests_passed += 1

            if not forward_results.get('error'):
                total_tests += 1
                if forward_results['passed']:
                    tests_passed += 1

            if not layer_results.get('error'):
                all_layers_passed = all(
                    r.get('passed', False) for r in layer_results.values()
                    if not r.get('error')
                )
                total_tests += 1
                if all_layers_passed:
                    tests_passed += 1

            all_batches_passed = all(
                r.get('passed', False) for r in batch_results.values()
                if not r.get('error')
            )
            total_tests += 1
            if all_batches_passed:
                tests_passed += 1

            print(f"  Tests Passed: {tests_passed}/{total_tests}")
            print("=" * 80)

        return all_results


# Example usage with proper weight transmission
def example_test():
    """
    Example: Test SwinTransformer implementations with proper weight transmission
    """

    print("\n" + "="*80)
    print("EXAMPLE: SwinTransformer Test - Tiny Configuration")
    print("="*80)

    # Tiny configuration
    config = {
        'img_size': 64, # Reduced
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 10, # Reduced
        'embed_dim': 48, # Reduced
        'depths': [1, 1, 1], # Reduced
        'num_heads': [2, 4, 8], # Reduced
        'window_size': 4, # Reduced
        'mlp_ratio': 2.0 # Reduced
    }

    # Create tester
    tester = SwinTransformerTester(**config)

    # Create PyTorch model
    pytorch_model = SwinTransformerAPI(**config, qkv_bias=True)

    # Extract weights
    print("\nExtracting weights from PyTorch model...")
    weights = tester.extract_all_weights(pytorch_model)
    print(f"  - Extracted {len(weights['layers'])} layers")
    for i, layer in enumerate(weights['layers']):
        print(f"  - Layer {i}: {len(layer['blocks'])} blocks, downsample={'Yes' if layer['downsample'] else 'No'}")

    # Create manual model with extracted weights
    print("\nCreating manual model with extracted weights...")
    manual_model = ManualSwinTransformer(**config, weights=weights)

    # Run tests
    print("\nRunning tests...")
    results = tester.run_all_tests(pytorch_model, manual_model)

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    print("SwinTransformer Weight Transmission Test")
    print("=" * 80)
    example_test()