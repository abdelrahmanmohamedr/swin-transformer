############################################################################################################
# CONSTANT INITIALIZATION FUNCTION IMPLEMENTATION (PyTorch vs NumPy)
# ----------------------------------------------------------------------------------------------------------
# This script demonstrates the difference between the original PyTorch implementation of `constant_` used 
# in neural network initialization (`torch.nn.init.constant_`) and a customized NumPy version that replicates 
# the same behavior. The goal is to fill a tensor/array with a given constant value.
############################################################################################################

import numpy as np
import torch
import torch as nn


# ==========================================================================================================
# ORIGINAL PYTORCH IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
"""
# def constant_(tensor: Tensor, val: float) -> Tensor:
#    """ r"""Fill the input Tensor with the value :math:`\text{val}`.

#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         val: the value to fill the tensor with

#     Examples:
#         >>> w = torch.empty(3, 5)
#         >>> nn.init.constant_(w, 0.3)
#     """
#     if torch.overrides.has_torch_function_variadic(tensor):
#         return torch.overrides.handle_torch_function(
#             constant_, (tensor,), tensor=tensor, val=val
#         )
#     return _no_grad_fill_(tensor, val)
# """
# ==========================================================================================================


############################################################################################################
# CONSTANT INITIALIZATION FUNCTION IMPLEMENTATION (PyTorch vs NumPy)
############################################################################################################

import numpy as np
import torch
import torch.nn as nn


# ==========================================================================================================
# NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
def constant_(array, val):
    """
    Fill the given NumPy array with the constant value 'val' (in-place).

    Args:
        array (np.ndarray): the input array (any shape)
        val (float): the value to fill the array with

    Returns:
        np.ndarray: the same array filled with val
    """
    array.fill(val)
    return array
# ==========================================================================================================


# ==========================================================================================================
# TEST CASES - Compare NumPy vs PyTorch
# ----------------------------------------------------------------------------------------------------------
def test_constant_init():
    """
    Compare the NumPy constant_ function against PyTorch's nn.init.constant_
    """
    print("="*80)
    print("COMPARING NumPy constant_ vs PyTorch nn.init.constant_")
    print("="*80)
    
    test_cases = [
        ("1D Array", (10,)),
        ("2D Square Matrix", (5, 5)),
        ("2D Non-Square Matrix", (3, 7)),
        ("3D Tensor", (2, 4, 3)),
        ("4D Tensor", (2, 3, 4, 5)),
        ("Single Element", (1,)),
        ("Large Matrix", (100, 100)),
    ]
    
    val = 0.42  # Test value
    all_passed = True
    
    for test_name, shape in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test: {test_name}")
        print(f"Shape: {shape}, Fill value: {val}")
        print(f"{'-'*80}")
        
        # NumPy version
        np_array = np.empty(shape)
        constant_(np_array, val)
        
        # PyTorch version
        torch_tensor = torch.empty(shape)
        nn.init.constant_(torch_tensor, val)
        torch_array = torch_tensor.numpy()
        
        # Compare results
        match = np.allclose(np_array, torch_array)
        all_equal_to_val = np.all(np_array == val)
        max_diff = np.abs(np_array - torch_array).max()
        
        print(f"NumPy - All values equal to {val}: {all_equal_to_val}")
        print(f"PyTorch - All values equal to {val}: {np.all(torch_array == val)}")
        print(f"Arrays match: {match}")
        print(f"Max difference: {max_diff}")
        print(f"Sample values (NumPy): {np_array.flat[:5]}")
        print(f"Sample values (PyTorch): {torch_array.flat[:5]}")
        
        if match and all_equal_to_val:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - NumPy implementation matches PyTorch!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return all_passed


# ==========================================================================================================
# USAGE EXAMPLES
# ----------------------------------------------------------------------------------------------------------
def usage_examples():
    """Show usage examples comparing both implementations"""
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    # Example 1: Basic 2D array
    print("\nExample 1: Fill 2D array with 0.3")
    print("-"*40)
    
    # NumPy
    np_w = np.empty((3, 4))
    print("NumPy - Before:\n", np_w)
    constant_(np_w, 0.3)
    print("NumPy - After:\n", np_w)
    
    # PyTorch
    torch_w = torch.empty((3, 4))
    print("\nPyTorch - Before:\n", torch_w)
    nn.init.constant_(torch_w, 0.3)
    print("PyTorch - After:\n", torch_w)
    
    print(f"\nResults match: {np.allclose(np_w, torch_w.numpy())}")
    
    # Example 2: 3D tensor
    print("\n" + "="*80)
    print("Example 2: Fill 3D tensor with -1.0")
    print("-"*40)
    
    # NumPy
    np_tensor = np.empty((2, 3, 4))
    constant_(np_tensor, -1.0)
    print(f"NumPy - Shape: {np_tensor.shape}, All = -1.0: {np.all(np_tensor == -1.0)}")
    
    # PyTorch
    torch_tensor = torch.empty((2, 3, 4))
    nn.init.constant_(torch_tensor, -1.0)
    print(f"PyTorch - Shape: {tuple(torch_tensor.shape)}, All = -1.0: {torch.all(torch_tensor == -1.0)}")
    
    print(f"Results match: {np.allclose(np_tensor, torch_tensor.numpy())}")
    
    # Example 3: Initialize bias to zero
    print("\n" + "="*80)
    print("Example 3: Initialize bias vector to zero")
    print("-"*40)
    
    # NumPy
    np_bias = np.empty(10)
    constant_(np_bias, 0.0)
    print(f"NumPy bias: {np_bias}")
    
    # PyTorch
    torch_bias = torch.empty(10)
    nn.init.constant_(torch_bias, 0.0)
    print(f"PyTorch bias: {torch_bias.numpy()}")
    
    print(f"Results match: {np.allclose(np_bias, torch_bias.numpy())}")


# ==========================================================================================================
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run comprehensive comparison tests
    test_constant_init()
    
    # Show usage examples
    usage_examples()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The NumPy implementation using array.fill(val) perfectly replicates")
    print("PyTorch's nn.init.constant_() behavior for all tensor shapes.")
    print("="*80)