"""
Classification-Focused Test Suite for Swin Transformer
Tests accuracy and classification agreement between PyTorch and Manual implementations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import time


class ClassificationTestSuite:
    """
    Test framework focused on classification accuracy and agreement
    between PyTorch and Manual Swin Transformer implementations
    """
    
    def __init__(self, num_classes=10):
        self.test_results = []
        self.num_classes = num_classes
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results with clear pass/fail status"""
        status = "✓ PASSED" if passed else "✗ FAILED"
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })
        print(f"{status}: {test_name}")
        if details:
            print(f"  {details}")
    
    def test_single_image_classification(self, pytorch_model, manual_model, test_input):
        """
        Test 1: Basic single batch classification agreement
        CORE TEST - Must pass for models to be equivalent
        """
        print("\n" + "="*80)
        print("TEST 1: Single Batch Classification Agreement")
        print("="*80)
        
        batch_size = test_input.shape[0]
        
        # Get predictions from both models
        with torch.no_grad():
            pytorch_model.eval()
            pt_output = pytorch_model(test_input)
        
        manual_output = manual_model(test_input)
        
        # Convert to tensor if needed
        if isinstance(manual_output, np.ndarray):
            manual_output = torch.from_numpy(manual_output)
        
        # Get predicted classes
        pt_predictions = torch.argmax(pt_output, dim=1)
        manual_predictions = torch.argmax(manual_output, dim=1)
        
        # Calculate agreement
        matches = (pt_predictions == manual_predictions).sum().item()
        agreement_rate = matches / batch_size
        
        # Detailed per-image analysis
        details_lines = [f"\nClassification Results (Batch Size: {batch_size}):"]
        for i in range(batch_size):
            pt_class = pt_predictions[i].item()
            manual_class = manual_predictions[i].item()
            match = "✓ Match" if pt_class == manual_class else "✗ Mismatch"
            
            # Get top-3 predictions for each model
            pt_probs = torch.softmax(pt_output[i], dim=0)
            manual_probs = torch.softmax(manual_output[i], dim=0)
            
            pt_top3 = torch.topk(pt_probs, min(3, self.num_classes))
            manual_top3 = torch.topk(manual_probs, min(3, self.num_classes))
            
            details_lines.append(f"\n  Image {i+1}: {match}")
            details_lines.append(f"    PyTorch Top-3: {[(pt_top3.indices[j].item(), f'{pt_top3.values[j].item():.4f}') for j in range(len(pt_top3.indices))]}")
            details_lines.append(f"    Manual  Top-3: {[(manual_top3.indices[j].item(), f'{manual_top3.values[j].item():.4f}') for j in range(len(manual_top3.indices))]}")
        
        details_lines.append(f"\n  Agreement: {matches}/{batch_size} ({agreement_rate*100:.1f}%)")
        details = "\n".join(details_lines)
        
        passed = agreement_rate == 1.0
        self.log_test("Single Batch Classification Agreement", passed, details)
        
        return passed, agreement_rate
    
    def test_classification_consistency(self, pytorch_model, manual_model, 
                                       num_samples=20, batch_size=4, img_size=224):
        """
        Test 2: Classification consistency across multiple random inputs
        Tests robustness and reliability of classification agreement
        """
        print("\n" + "="*80)
        print(f"TEST 2: Classification Consistency ({num_samples} samples)")
        print("="*80)
        
        total_images = 0
        total_matches = 0
        mismatched_batches = []
        
        num_batches = num_samples // batch_size
        
        for batch_idx in range(num_batches):
            # Generate random input
            test_input = torch.randn(batch_size, 3, img_size, img_size)
            
            # Get predictions
            with torch.no_grad():
                pytorch_model.eval()
                pt_output = pytorch_model(test_input)
            
            manual_output = manual_model(test_input)
            
            if isinstance(manual_output, np.ndarray):
                manual_output = torch.from_numpy(manual_output)
            
            pt_predictions = torch.argmax(pt_output, dim=1)
            manual_predictions = torch.argmax(manual_output, dim=1)
            
            # Count matches in this batch
            batch_matches = (pt_predictions == manual_predictions).sum().item()
            total_matches += batch_matches
            total_images += batch_size
            
            # Track mismatched batches
            if batch_matches < batch_size:
                mismatched_batches.append({
                    'batch_idx': batch_idx,
                    'pytorch': pt_predictions.tolist(),
                    'manual': manual_predictions.tolist(),
                    'matches': batch_matches
                })
        
        consistency_rate = total_matches / total_images
        
        details_lines = [f"\n  Total Images Tested: {total_images}"]
        details_lines.append(f"  Matching Predictions: {total_matches}")
        details_lines.append(f"  Consistency Rate: {consistency_rate*100:.2f}%")
        
        if mismatched_batches:
            details_lines.append(f"\n  Mismatched Batches: {len(mismatched_batches)}/{num_batches}")
            for mm in mismatched_batches[:3]:  # Show first 3 mismatches
                details_lines.append(f"    Batch {mm['batch_idx']}: PyTorch={mm['pytorch']}, Manual={mm['manual']}")
        
        details = "\n".join(details_lines)
        
        passed = consistency_rate == 1.0
        self.log_test(f"Classification Consistency ({num_samples} samples)", passed, details)
        
        return passed, consistency_rate
    
    def test_class_distribution_agreement(self, pytorch_model, manual_model, 
                                         num_samples=100, batch_size=10, img_size=224):
        """
        Test 3: Class distribution agreement
        Tests if both models predict similar class distributions
        """
        print("\n" + "="*80)
        print(f"TEST 3: Class Distribution Agreement ({num_samples} samples)")
        print("="*80)
        
        pt_class_counts = np.zeros(self.num_classes)
        manual_class_counts = np.zeros(self.num_classes)
        
        num_batches = num_samples // batch_size
        
        for _ in range(num_batches):
            test_input = torch.randn(batch_size, 3, img_size, img_size)
            
            with torch.no_grad():
                pytorch_model.eval()
                pt_output = pytorch_model(test_input)
            
            manual_output = manual_model(test_input)
            
            if isinstance(manual_output, np.ndarray):
                manual_output = torch.from_numpy(manual_output)
            
            pt_predictions = torch.argmax(pt_output, dim=1)
            manual_predictions = torch.argmax(manual_output, dim=1)
            
            # Count class occurrences
            for c in range(self.num_classes):
                pt_class_counts[c] += (pt_predictions == c).sum().item()
                manual_class_counts[c] += (manual_predictions == c).sum().item()
        
        # Calculate distribution similarity
        pt_dist = pt_class_counts / num_samples
        manual_dist = manual_class_counts / num_samples
        
        # Check if distributions match exactly
        distributions_match = np.array_equal(pt_class_counts, manual_class_counts)
        
        # Calculate distribution difference (for info)
        dist_diff = np.abs(pt_dist - manual_dist)
        max_diff = np.max(dist_diff)
        
        details_lines = ["\n  Class Distribution Comparison:"]
        details_lines.append(f"  {'Class':<8} {'PyTorch':<10} {'Manual':<10} {'Diff':<10}")
        details_lines.append("  " + "-"*40)
        for c in range(self.num_classes):
            details_lines.append(
                f"  {c:<8} {pt_class_counts[c]:<10.0f} {manual_class_counts[c]:<10.0f} "
                f"{abs(pt_class_counts[c] - manual_class_counts[c]):<10.0f}"
            )
        details_lines.append(f"\n  Max Distribution Difference: {max_diff:.4f}")
        details_lines.append(f"  Distributions Identical: {distributions_match}")
        
        details = "\n".join(details_lines)
        
        self.log_test("Class Distribution Agreement", distributions_match, details)
        
        return distributions_match, max_diff
    
    def test_batch_size_invariance(self, pytorch_model, manual_model, 
                                   batch_sizes=[1, 2, 4, 8, 16], img_size=224):
        """
        Test 4: Classification agreement across different batch sizes
        Ensures batch size doesn't affect classification agreement
        """
        print("\n" + "="*80)
        print("TEST 4: Batch Size Invariance")
        print("="*80)
        
        all_passed = True
        results = []
        
        for bs in batch_sizes:
            test_input = torch.randn(bs, 3, img_size, img_size)
            
            with torch.no_grad():
                pytorch_model.eval()
                pt_output = pytorch_model(test_input)
            
            manual_output = manual_model(test_input)
            
            if isinstance(manual_output, np.ndarray):
                manual_output = torch.from_numpy(manual_output)
            
            pt_predictions = torch.argmax(pt_output, dim=1)
            manual_predictions = torch.argmax(manual_output, dim=1)
            
            matches = (pt_predictions == manual_predictions).sum().item()
            agreement = matches / bs
            batch_passed = agreement == 1.0
            
            if not batch_passed:
                all_passed = False
                details = f"Agreement: {matches}/{bs} ({agreement*100:.1f}%), PyTorch: {pt_predictions.tolist()}, Manual: {manual_predictions.tolist()}"
            else:
                details = f"Agreement: {matches}/{bs} (100%)"
            
            results.append((bs, batch_passed, details))
            self.log_test(f"Batch Size {bs}", batch_passed, details)
        
        return all_passed
    
    def test_deterministic_behavior(self, pytorch_model, manual_model, 
                                   num_runs=5, seed=42, img_size=224):
        """
        Test 5: Deterministic behavior
        Tests if both models produce consistent results with same input
        """
        print("\n" + "="*80)
        print(f"TEST 5: Deterministic Behavior ({num_runs} runs)")
        print("="*80)
        
        # Set seed and create fixed input
        torch.manual_seed(seed)
        test_input = torch.randn(2, 3, img_size, img_size)
        
        pt_predictions_list = []
        manual_predictions_list = []
        
        for run in range(num_runs):
            with torch.no_grad():
                pytorch_model.eval()
                pt_output = pytorch_model(test_input)
            
            manual_output = manual_model(test_input)
            
            if isinstance(manual_output, np.ndarray):
                manual_output = torch.from_numpy(manual_output)
            
            pt_predictions = torch.argmax(pt_output, dim=1)
            manual_predictions = torch.argmax(manual_output, dim=1)
            
            pt_predictions_list.append(pt_predictions)
            manual_predictions_list.append(manual_predictions)
        
        # Check if PyTorch is deterministic
        pt_deterministic = all(torch.equal(pt_predictions_list[0], pred) 
                              for pred in pt_predictions_list)
        
        # Check if Manual is deterministic
        manual_deterministic = all(torch.equal(manual_predictions_list[0], pred) 
                                  for pred in manual_predictions_list)
        
        # Check if they agree across all runs
        all_agree = all(torch.equal(pt_predictions_list[i], manual_predictions_list[i]) 
                       for i in range(num_runs))
        
        details_lines = [f"\n  PyTorch Deterministic: {pt_deterministic}"]
        details_lines.append(f"  Manual Deterministic: {manual_deterministic}")
        details_lines.append(f"  Cross-Model Agreement (all runs): {all_agree}")
        
        if not all_agree:
            details_lines.append("\n  Run-by-run comparison:")
            for i in range(num_runs):
                details_lines.append(f"    Run {i+1}: PyTorch={pt_predictions_list[i].tolist()}, Manual={manual_predictions_list[i].tolist()}")
        
        details = "\n".join(details_lines)
        
        passed = pt_deterministic and manual_deterministic and all_agree
        self.log_test("Deterministic Behavior", passed, details)
        
        return passed
    
    def test_extreme_input_classification(self, pytorch_model, manual_model, img_size=224):
        """
        Test 6: Classification on edge case inputs
        Tests model robustness on unusual inputs
        """
        print("\n" + "="*80)
        print("TEST 6: Edge Case Input Classification")
        print("="*80)
        
        test_cases = {
            "All Zeros": torch.zeros(1, 3, img_size, img_size),
            "All Ones": torch.ones(1, 3, img_size, img_size),
            "Very Large Values": torch.randn(1, 3, img_size, img_size) * 1000,
            "Very Small Values": torch.randn(1, 3, img_size, img_size) * 0.001,
            "Uniform Random [0,1]": torch.rand(1, 3, img_size, img_size),
            "Negative Values": -torch.abs(torch.randn(1, 3, img_size, img_size)),
        }
        
        all_passed = True
        results = []
        
        for case_name, test_input in test_cases.items():
            with torch.no_grad():
                pytorch_model.eval()
                pt_output = pytorch_model(test_input)
            
            manual_output = manual_model(test_input)
            
            if isinstance(manual_output, np.ndarray):
                manual_output = torch.from_numpy(manual_output)
            
            pt_predictions = torch.argmax(pt_output, dim=1)
            manual_predictions = torch.argmax(manual_output, dim=1)
            
            case_passed = torch.equal(pt_predictions, manual_predictions)
            
            if not case_passed:
                all_passed = False
                details = f"PyTorch: {pt_predictions.tolist()}, Manual: {manual_predictions.tolist()}"
            else:
                details = "Predictions match"
            
            self.log_test(f"Edge Case: {case_name}", case_passed, details)
            results.append((case_name, case_passed))
        
        return all_passed
    
    def test_confidence_agreement(self, pytorch_model, manual_model, 
                                 test_input, confidence_threshold=0.1):
        """
        Test 7: Confidence score agreement (optional, informational)
        Checks if confidence scores are similar when classes match
        """
        print("\n" + "="*80)
        print("TEST 7: Confidence Score Analysis (Informational)")
        print("="*80)
        
        batch_size = test_input.shape[0]
        
        with torch.no_grad():
            pytorch_model.eval()
            pt_output = pytorch_model(test_input)
        
        manual_output = manual_model(test_input)
        
        if isinstance(manual_output, np.ndarray):
            manual_output = torch.from_numpy(manual_output)
        
        pt_predictions = torch.argmax(pt_output, dim=1)
        manual_predictions = torch.argmax(manual_output, dim=1)
        
        pt_probs = torch.softmax(pt_output, dim=1)
        manual_probs = torch.softmax(manual_output, dim=1)
        
        details_lines = ["\n  Confidence Score Comparison:"]
        high_confidence_agreement = 0
        
        for i in range(batch_size):
            pt_class = pt_predictions[i].item()
            manual_class = manual_predictions[i].item()
            
            pt_conf = pt_probs[i, pt_class].item()
            manual_conf = manual_probs[i, manual_class].item()
            
            conf_diff = abs(pt_conf - manual_conf)
            
            if pt_class == manual_class:
                if conf_diff < confidence_threshold:
                    high_confidence_agreement += 1
                    status = "✓"
                else:
                    status = "~"
            else:
                status = "✗"
            
            details_lines.append(
                f"  Image {i+1} {status}: Class {pt_class}/{manual_class}, "
                f"Confidence {pt_conf:.4f}/{manual_conf:.4f}, Diff: {conf_diff:.4f}"
            )
        
        details_lines.append(f"\n  High Confidence Agreement: {high_confidence_agreement}/{batch_size}")
        details = "\n".join(details_lines)
        
        # This test is informational, always "passes"
        self.log_test("Confidence Score Analysis", True, details)
        
        return True
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("FINAL CLASSIFICATION TEST REPORT")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ({success_rate:.1f}%)")
        print(f"   Failed: {failed_tests}")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"   • {result['name']}")
        
        if failed_tests == 0:
            print("\n" + "="*80)
            print("🎉 ALL TESTS PASSED!")
            print("="*80)
            print("✓ Both models classify images identically")
            print("✓ Classification is consistent across different inputs")
            print("✓ Models are functionally equivalent for classification tasks")
        else:
            print("\n" + "="*80)
            print("⚠️  SOME TESTS FAILED")
            print("="*80)
            print("The models do not produce identical classifications.")
            print("Review the failed tests above for details.")
        
        print("\n" + "="*80)
        return passed_tests == total_tests


def run_classification_test_suite(pytorch_model, manual_model, num_classes=10):
    """
    Main function to run classification-focused test suite
    
    Args:
        pytorch_model: Original PyTorch Swin Transformer
        manual_model: Manual implementation of Swin Transformer
        num_classes: Number of output classes
    
    Returns:
        bool: True if all tests pass
    """
    print("="*80)
    print("SWIN TRANSFORMER - CLASSIFICATION TEST SUITE")
    print("="*80)
    print("\nFocus: Classification Accuracy and Agreement")
    print("Goal: Verify both models produce identical class predictions\n")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test suite
    test_suite = ClassificationTestSuite(num_classes=num_classes)
    
    # Create test input
    test_input = torch.randn(1, 3, 56, 56)
    
    print("Running tests...\n")
    
    # Run all classification tests
    test_suite.test_single_image_classification(pytorch_model, manual_model, test_input)
    test_suite.test_classification_consistency(pytorch_model, manual_model, num_samples=10, batch_size=2, img_size=56)
    test_suite.test_class_distribution_agreement(pytorch_model, manual_model, num_samples=50, batch_size=5, img_size=56)
    test_suite.test_batch_size_invariance(pytorch_model, manual_model, batch_sizes=[1, 2, 4], img_size=56)
    test_suite.test_deterministic_behavior(pytorch_model, manual_model, num_runs=3, img_size=56)
    test_suite.test_extreme_input_classification(pytorch_model, manual_model, img_size=56)
    test_suite.test_confidence_agreement(pytorch_model, manual_model, test_input)
    
    # Generate final report
    all_passed = test_suite.generate_report()
    
    return all_passed


if __name__ == "__main__":
    # Example usage (uncomment when models are ready):
    from modified_swin_test import ManualSwinTransformer
    from weight_extraction_helper import (
        extract_all_weights_from_pytorch,
        create_small_swin_pytorch,
        create_small_swin_manual,
        verify_weight_shapes,
        print_weight_statistics
    )
    
    pytorch_model = create_small_swin_pytorch()
    weights = extract_all_weights_from_pytorch(pytorch_model)
    manual_model = create_small_swin_manual(weights=weights)
    
    success = run_classification_test_suite(pytorch_model, manual_model, num_classes=10)
    exit(0 if success else 1)
    
    print("Please import your models and call run_classification_test_suite()")