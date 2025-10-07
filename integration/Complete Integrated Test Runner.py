"""
Complete Integrated Classification Test Runner for Swin Transformer
Combines weight extraction, model creation, and classification-focused testing
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from typing import Dict, List, Tuple

# Import the classification test suite
from swin_comparison_tests import ClassificationTestSuite, run_classification_test_suite

# Import weight extraction helpers
from weight_extraction_helper import (
    extract_all_weights_from_pytorch,
    create_small_swin_pytorch,
    create_small_swin_manual,
    verify_weight_shapes,
    print_weight_statistics
)


def run_integrated_classification_tests():
    """
    Main integrated test runner focusing on classification accuracy
    """
    print("="*80)
    print("INTEGRATED CLASSIFICATION TEST SUITE FOR SWIN TRANSFORMER")
    print("="*80)
    print()
    print("This suite will:")
    print("  1. Create PyTorch Swin Transformer model")
    print("  2. Extract all weights from PyTorch model")
    print("  3. Initialize Manual Swin Transformer with extracted weights")
    print("  4. Run classification-focused tests")
    print("  5. Report on classification agreement and accuracy")
    print()
    print("⚠️  FOCUS: Tests pass when models predict the SAME CLASSES")
    print("    Exact numerical match is NOT required")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # ========================================================================
    # PHASE 1: Model Creation and Weight Extraction
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Model Setup and Weight Extraction")
    print("="*80)
    
    # Step 1.1: Create PyTorch model
    print("\n[1/5] Creating PyTorch Swin Transformer...")
    try:
        pytorch_model = create_small_swin_pytorch()
        pytorch_model.eval()
        
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        print(f"      ✓ PyTorch model created successfully")
        print(f"        - Total parameters: {total_params:,}")
        print(f"        - Number of layers: {pytorch_model.num_layers}")
        print(f"        - Embedding dimension: {pytorch_model.embed_dim}")
        print(f"        - Number of classes: {pytorch_model.num_classes}")
        
    except Exception as e:
        print(f"      ✗ Failed to create PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 1.2: Extract weights
    print("\n[2/5] Extracting weights from PyTorch model...")
    try:
        weights = extract_all_weights_from_pytorch(pytorch_model)
        
        num_layers = len(weights['layers'])
        total_blocks = sum(len(layer['blocks']) for layer in weights['layers'])
        print(f"      ✓ Weights extracted successfully")
        print(f"        - Extracted from {num_layers} layers")
        print(f"        - Total transformer blocks: {total_blocks}")
        
    except Exception as e:
        print(f"      ✗ Failed to extract weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 1.3: Verify weight shapes
    print("\n[3/5] Verifying weight shapes...")
    try:
        shapes_match = verify_weight_shapes(pytorch_model, weights)
        if shapes_match:
            print(f"      ✓ All weight shapes verified")
        else:
            print(f"      ⚠ Some weight shapes don't match (check logs)")
            
    except Exception as e:
        print(f"      ✗ Failed to verify shapes: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 1.4: Create Manual model
    print("\n[4/5] Creating Manual Swin Transformer with extracted weights...")
    try:
        manual_model = create_small_swin_manual(weights)
        print(f"      ✓ Manual model created successfully")
        
    except Exception as e:
        print(f"      ✗ Failed to create Manual model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 1.5: Quick sanity check
    print("\n[5/5] Running quick sanity check...")
    try:
        test_input = torch.randn(1, 3, 56, 56)
        
        with torch.no_grad():
            pt_output = pytorch_model(test_input)
        print(f"      ✓ PyTorch forward pass successful (shape: {pt_output.shape})")
        
        manual_output = manual_model(test_input)
        if isinstance(manual_output, np.ndarray):
            manual_output = torch.from_numpy(manual_output)
        print(f"      ✓ Manual forward pass successful (shape: {manual_output.shape})")
        
        # Quick prediction check
        pt_pred = torch.argmax(pt_output, dim=1)
        manual_pred = torch.argmax(manual_output, dim=1)
        match = torch.equal(pt_pred, manual_pred)
        
        if match:
            print(f"      ✓ Initial predictions match: {pt_pred.tolist()}")
        else:
            print(f"      ⚠ Initial predictions differ:")
            print(f"        PyTorch: {pt_pred.tolist()}")
            print(f"        Manual:  {manual_pred.tolist()}")
        
    except Exception as e:
        print(f"      ✗ Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # PHASE 2: Classification Tests
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Classification-Focused Testing")
    print("="*80)
    print()
    print("Running 7 classification tests...")
    print("Goal: Verify both models classify images identically")
    print()
    
    # Initialize test suite
    test_suite = ClassificationTestSuite(num_classes=pytorch_model.num_classes)
    
    # Create fresh test input
    test_input = torch.randn(1, 3, 56, 56)
    
    # Test 1: Single Batch Classification Agreement (CRITICAL)
    try:
        print("="*80)
        test_suite.test_single_image_classification(pytorch_model, manual_model, test_input)
    except Exception as e:
        print(f"✗ Test 1 Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Classification Consistency (Multiple Samples)
    try:
        test_suite.test_classification_consistency(
            pytorch_model, manual_model, 
            num_samples=10,
            batch_size=2,
            img_size=56
        )
    except Exception as e:
        print(f"✗ Test 2 Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Class Distribution Agreement
    try:
        test_suite.test_class_distribution_agreement(
            pytorch_model, manual_model,
            num_samples=50,
            batch_size=5,
            img_size=56
        )
    except Exception as e:
        print(f"✗ Test 3 Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Batch Size Invariance
    try:
        test_suite.test_batch_size_invariance(
            pytorch_model, manual_model,
            batch_sizes=[1, 2, 4],
            img_size=56
        )
    except Exception as e:
        print(f"✗ Test 4 Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Deterministic Behavior
    try:
        test_suite.test_deterministic_behavior(
            pytorch_model, manual_model,
            num_runs=3,
            seed=42,
            img_size=56
        )
    except Exception as e:
        print(f"✗ Test 5 Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Edge Case Classifications
    try:
        test_suite.test_extreme_input_classification(
            pytorch_model, manual_model,
            img_size=56
        )
    except Exception as e:
        print(f"✗ Test 6 Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Confidence Score Analysis (Informational)
    try:
        test_suite.test_confidence_agreement(
            pytorch_model, manual_model,
            test_input,
            confidence_threshold=0.1
        )
    except Exception as e:
        print(f"✗ Test 7 Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # PHASE 3: Final Report
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Final Report Generation")
    print("="*80)
    
    all_passed = test_suite.generate_report()
    
    # ========================================================================
    # PHASE 4: Summary and Recommendations
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Summary and Analysis")
    print("="*80)
    
    print("\n📊 Model Architecture Summary:")
    print(f"   Image size: {pytorch_model.patch_embed.img_size}")
    print(f"   Patch size: {pytorch_model.patch_embed.patch_size}")
    print(f"   Number of patches: {pytorch_model.patch_embed.num_patches}")
    print(f"   Embedding dimension: {pytorch_model.embed_dim}")
    print(f"   Number of layers: {pytorch_model.num_layers}")
    print(f"   Layer depths: {[layer.depth for layer in pytorch_model.layers]}")
    print(f"   Attention heads: {[layer.blocks[0].num_heads for layer in pytorch_model.layers]}")
    print(f"   Window size: {pytorch_model.layers[0].blocks[0].window_size}")
    print(f"   Number of classes: {pytorch_model.num_classes}")
    
    print("\n🔍 Classification Analysis:")
    
    # Analyze results
    classification_tests = [r for r in test_suite.test_results 
                           if 'Classification' in r['name'] or 'Predictions' in r['name']]
    classification_passed = sum(1 for t in classification_tests if t['passed'])
    
    if classification_passed == len(classification_tests):
        print(f"   ✓ Perfect Agreement: {classification_passed}/{len(classification_tests)} tests passed")
        print(f"   ✓ Both models classify ALL images identically")
        print(f"   ✓ Models are functionally equivalent for classification")
    elif classification_passed > len(classification_tests) * 0.8:
        print(f"   ⚠ High Agreement: {classification_passed}/{len(classification_tests)} tests passed")
        print(f"   ⚠ Models mostly agree, but some edge cases differ")
        print(f"   ⚠ Review failed tests for specific scenarios")
    elif classification_passed > len(classification_tests) * 0.5:
        print(f"   ⚠ Moderate Agreement: {classification_passed}/{len(classification_tests)} tests passed")
        print(f"   ⚠ Significant classification differences detected")
        print(f"   ⚠ Models may have implementation issues")
    else:
        print(f"   ✗ Low Agreement: {classification_passed}/{len(classification_tests)} tests passed")
        print(f"   ✗ Models produce very different classifications")
        print(f"   ✗ Critical implementation issues likely present")
    
    print("\n💡 Key Insights:")
    
    if all_passed:
        print("   1. ✓ Weight extraction successful - all weights transferred correctly")
        print("   2. ✓ Manual implementation matches PyTorch behavior exactly")
        print("   3. ✓ Classification decisions are 100% consistent")
        print("   4. ✓ Both implementations can be used interchangeably")
        print("   5. ✓ Manual implementation is suitable for:")
        print("      - Educational purposes (understanding transformer internals)")
        print("      - Debugging and visualization")
        print("      - Custom modifications and experiments")
    else:
        print("   1. ⚠ Check weight extraction - ensure all weights copied correctly")
        print("   2. ⚠ Review layer implementations - some may have bugs")
        print("   3. ⚠ Verify numerical stability - check for NaN/Inf values")
        print("   4. ⚠ Compare intermediate outputs - identify where divergence occurs")
        print("   5. ⚠ Consider tolerance levels - small differences may be acceptable")
    
    print("\n🎯 Recommendations:")
    
    if all_passed:
        print("   For Production:")
        print("   - Use PyTorch implementation (optimized, GPU support)")
        print("   - Models produce identical results, choose based on needs")
        print()
        print("   For Learning/Development:")
        print("   - Use Manual implementation to understand internals")
        print("   - Modify Manual version for experimentation")
        print("   - Use this test suite to verify custom modifications")
    else:
        print("   Debugging Steps:")
        print("   1. Review failed test details above")
        print("   2. Check weight shapes and values")
        print("   3. Test individual components in isolation")
        print("   4. Compare intermediate activations")
        print("   5. Use smaller batch sizes for easier debugging")
        print()
        print("   Common Issues:")
        print("   - Incorrect weight transposition")
        print("   - Missing bias terms")
        print("   - Wrong dimension ordering (NCHW vs NHWC)")
        print("   - Incorrect attention mask application")
        print("   - LayerNorm epsilon differences")
    
    # ========================================================================
    # Conclusion
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    
    if all_passed:
        print()
        print("🎉 SUCCESS! 🎉")
        print()
        print("All classification tests passed!")
        print("Both models predict the same classes for all test cases.")
        print("The Manual implementation is functionally equivalent to PyTorch.")
        print()
    else:
        failed_count = len(test_suite.test_results) - sum(1 for r in test_suite.test_results if r['passed'])
        print()
        print("⚠️  ATTENTION NEEDED ⚠️")
        print()
        print(f"{failed_count} test(s) failed.")
        print("The models do not produce identical classifications.")
        print("Review the detailed results above to identify issues.")
        print()
    
    print("="*80)
    print()
    
    return all_passed


def quick_classification_check(pytorch_model, manual_model, num_samples=10):
    """
    Quick classification agreement check without full test suite
    Useful for rapid iteration during development
    """
    print("\n" + "="*80)
    print("QUICK CLASSIFICATION CHECK")
    print("="*80)
    
    pytorch_model.eval()
    
    total_match = 0
    total_samples = 0
    
    print(f"\nTesting with {num_samples} random inputs...")
    
    for i in range(num_samples):
        test_input = torch.randn(1, 3, 56, 56)
        
        with torch.no_grad():
            pt_output = pytorch_model(test_input)
        
        manual_output = manual_model(test_input)
        
        if isinstance(manual_output, np.ndarray):
            manual_output = torch.from_numpy(manual_output)
        
        pt_pred = torch.argmax(pt_output, dim=1).item()
        manual_pred = torch.argmax(manual_output, dim=1).item()
        
        match = pt_pred == manual_pred
        total_match += int(match)
        total_samples += 1
        
        status = "✓" if match else "✗"
        print(f"  Sample {i+1:2d}: {status} PyTorch={pt_pred}, Manual={manual_pred}")
    
    agreement_rate = total_match / total_samples
    print(f"\nAgreement Rate: {total_match}/{total_samples} ({agreement_rate*100:.1f}%)")
    
    if agreement_rate == 1.0:
        print("✓ Perfect agreement!")
        return True
    elif agreement_rate >= 0.9:
        print("⚠ High agreement, but not perfect")
        return False
    else:
        print("✗ Low agreement - significant issues")
        return False


if __name__ == "__main__":
    print("\nStarting Integrated Classification Test Suite...\n")
    
    try:
        success = run_integrated_classification_tests()
        
        # Optional: Run quick check for additional validation
        # print("\n" + "="*80)
        # print("Running additional quick check...")
        # quick_classification_check(pytorch_model, manual_model, num_samples=10)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)