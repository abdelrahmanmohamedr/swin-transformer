############################################################################################################
# CONSTANT INITIALIZATION FUNCTION IMPLEMENTATION (PyTorch vs NumPy)
# ----------------------------------------------------------------------------------------------------------
# This script demonstrates the difference between the original PyTorch implementation of `constant_` used 
# in neural network initialization (`torch.nn.init.constant_`) and a customized NumPy version that replicates 
# the same behavior. The goal is to fill a tensor/array with a given constant value.
############################################################################################################


# ==========================================================================================================
# ORIGINAL PYTORCH IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
"""
def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fill the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            constant_, (tensor,), tensor=tensor, val=val
        )
    return _no_grad_fill_(tensor, val)
"""
# ==========================================================================================================


# ==========================================================================================================
# CUSTOMIZED NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
import numpy as np

def constant_(array: np.ndarray, val: float) -> np.ndarray:
    """
    Fill the given NumPy array with the constant value 'val'.

    Args:
        array (np.ndarray): the input array
        val (float): the value to fill the array with

    Returns:
        np.ndarray: the same array filled with val
    """
    array[:] = val   # inplace fill
    return array
# ==========================================================================================================


# ==========================================================================================================
# USAGE EXAMPLE
# ----------------------------------------------------------------------------------------------------------
# Step 1: Create an uninitialized NumPy array using `np.empty`
# Step 2: Apply the custom `constant_` function to fill it with the constant value
# Step 3: Print the array before and after filling
#############################################################

"""
w = np.empty((5, 5))
print("Before:")
print(w)

# fill with 0.3
constant_(w, 0.3)
print("After:")
print(w)
"""
# ==========================================================================================================


# ==========================================================================================================
# SAMPLE OUTPUT
# ----------------------------------------------------------------------------------------------------------
"""
Before:
[[ 8.49148011e-314  2.78136702e-309  7.29112202e-304  2.79003402e-308
   5.45352983e-312]
 [ 2.55053947e-313  4.31108272e-308  1.34497462e-284  1.15436571e-311
   4.31108272e-308]
 [ 1.34497462e-284  3.95919325e+020 -5.59825124e+093  4.94559711e-321
   9.45700527e-308]
 [ 0.00000000e+000  0.00000000e+000  0.00000000e+000  0.00000000e+000
   0.00000000e+000]
 [ 0.00000000e+000  1.22383816e-307  6.95214593e-310  1.39071615e-307
   0.00000000e+000]]
After:
[[0.3 0.3 0.3 0.3 0.3]
 [0.3 0.3 0.3 0.3 0.3]
 [0.3 0.3 0.3 0.3 0.3]
 [0.3 0.3 0.3 0.3 0.3]
 [0.3 0.3 0.3 0.3 0.3]]
"""
# ==========================================================================================================
