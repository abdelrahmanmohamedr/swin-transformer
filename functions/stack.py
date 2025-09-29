import numpy as np
import torch


class TorchStack:
    """
    Custom implementation of torch.stack() function.
    
    Stacks a sequence of tensors along a new dimension.
    All tensors must have the same shape.
    """
    
    @staticmethod
    def stack(tensors, dim=0):
        """
        Stack tensors along a new dimension.
        
        Args:
            tensors: List or tuple of arrays/tensors with the same shape
            dim: Dimension along which to insert the new axis (default: 0)
            
        Returns:
            Stacked array with one additional dimension
        """
        # Convert all inputs to numpy arrays
        arrays = [np.array(t) for t in tensors]
        
        # Verify all tensors have the same shape
        first_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != first_shape:
                raise ValueError(
                    f"All tensors must have the same shape. "
                    f"Tensor 0 has shape {first_shape}, "
                    f"but tensor {i} has shape {arr.shape}"
                )
        
        # Handle negative dimension indexing
        if dim < 0:
            dim = len(first_shape) + 1 + dim
        
        # Validate dimension range
        if dim < 0 or dim > len(first_shape):
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-len(first_shape)-1}, {len(first_shape)}], but got {dim})"
            )
        
        # Expand dimensions: add new axis at the specified position
        expanded = [np.expand_dims(arr, axis=dim) for arr in arrays]
        
        # Concatenate along the new dimension
        result = np.concatenate(expanded, axis=dim)
        
        return result


# Comparison with PyTorch
if __name__ == "__main__":
    print("="*70)
    print("Comparing Custom torch.stack vs PyTorch torch.stack")
    print("="*70 + "\n")
    
    # Example 1: Stack 1D tensors along dim=0
    print("Example 1: Stack 1D tensors (dim=0)")
    print("-" * 50)
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    
    custom_result = TorchStack.stack([a, b, c], dim=0)
    torch_result = torch.stack([torch.tensor(a), torch.tensor(b), torch.tensor(c)], dim=0).numpy()
    
    print(f"Input tensors:")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  c = {c}")
    print(f"\nCustom Output (shape {custom_result.shape}):")
    print(custom_result)
    print(f"\nPyTorch Output (shape {torch_result.shape}):")
    print(torch_result)
    print(f"\nDifference: {np.max(np.abs(custom_result - torch_result)):.2e}")
    print(f"Match: {np.allclose(custom_result, torch_result)}\n")
    
    # Example 2: Stack 1D tensors along dim=1
    print("Example 2: Stack 1D tensors (dim=1)")
    print("-" * 50)
    custom_result_dim1 = TorchStack.stack([a, b, c], dim=1)
    torch_result_dim1 = torch.stack([torch.tensor(a), torch.tensor(b), torch.tensor(c)], dim=1).numpy()
    
    print(f"Custom Output (shape {custom_result_dim1.shape}):")
    print(custom_result_dim1)
    print(f"\nPyTorch Output (shape {torch_result_dim1.shape}):")
    print(torch_result_dim1)
    print(f"\nDifference: {np.max(np.abs(custom_result_dim1 - torch_result_dim1)):.2e}")
    print(f"Match: {np.allclose(custom_result_dim1, torch_result_dim1)}\n")
    
    # Example 3: Stack 2D tensors along dim=0
    print("Example 3: Stack 2D tensors (dim=0)")
    print("-" * 50)
    x = [[1, 2, 3],
         [4, 5, 6]]
    y = [[7, 8, 9],
         [10, 11, 12]]
    
    custom_result_2d = TorchStack.stack([x, y], dim=0)
    torch_result_2d = torch.stack([torch.tensor(x), torch.tensor(y)], dim=0).numpy()
    
    print(f"Input tensors:")
    print(f"  x = \n{np.array(x)}")
    print(f"  y = \n{np.array(y)}")
    print(f"\nCustom Output (shape {custom_result_2d.shape}):")
    print(custom_result_2d)
    print(f"\nPyTorch Output (shape {torch_result_2d.shape}):")
    print(torch_result_2d)
    print(f"\nDifference: {np.max(np.abs(custom_result_2d - torch_result_2d)):.2e}")
    print(f"Match: {np.allclose(custom_result_2d, torch_result_2d)}\n")
    
    # Example 4: Stack 2D tensors along dim=1
    print("Example 4: Stack 2D tensors (dim=1)")
    print("-" * 50)
    custom_result_2d_dim1 = TorchStack.stack([x, y], dim=1)
    torch_result_2d_dim1 = torch.stack([torch.tensor(x), torch.tensor(y)], dim=1).numpy()
    
    print(f"Custom Output (shape {custom_result_2d_dim1.shape}):")
    print(custom_result_2d_dim1)
    print(f"\nPyTorch Output (shape {torch_result_2d_dim1.shape}):")
    print(torch_result_2d_dim1)
    print(f"\nDifference: {np.max(np.abs(custom_result_2d_dim1 - torch_result_2d_dim1)):.2e}")
    print(f"Match: {np.allclose(custom_result_2d_dim1, torch_result_2d_dim1)}\n")
    
    # Example 5: Stack 2D tensors along dim=2
    print("Example 5: Stack 2D tensors (dim=2)")
    print("-" * 50)
    custom_result_2d_dim2 = TorchStack.stack([x, y], dim=2)
    torch_result_2d_dim2 = torch.stack([torch.tensor(x), torch.tensor(y)], dim=2).numpy()
    
    print(f"Custom Output (shape {custom_result_2d_dim2.shape}):")
    print(custom_result_2d_dim2)
    print(f"\nPyTorch Output (shape {torch_result_2d_dim2.shape}):")
    print(torch_result_2d_dim2)
    print(f"\nDifference: {np.max(np.abs(custom_result_2d_dim2 - torch_result_2d_dim2)):.2e}")
    print(f"Match: {np.allclose(custom_result_2d_dim2, torch_result_2d_dim2)}\n")
    
    # Example 6: Negative dimension indexing
    print("Example 6: Stack with negative dim=-1 (last dimension)")
    print("-" * 50)
    custom_result_neg = TorchStack.stack([a, b, c], dim=-1)
    torch_result_neg = torch.stack([torch.tensor(a), torch.tensor(b), torch.tensor(c)], dim=-1).numpy()
    
    print(f"Custom Output (shape {custom_result_neg.shape}):")
    print(custom_result_neg)
    print(f"\nPyTorch Output (shape {torch_result_neg.shape}):")
    print(torch_result_neg)
    print(f"\nDifference: {np.max(np.abs(custom_result_neg - torch_result_neg)):.2e}")
    print(f"Match: {np.allclose(custom_result_neg, torch_result_neg)}\n")
    
    print("="*70)
    print("✓ All implementations match PyTorch!")
    print("="*70)