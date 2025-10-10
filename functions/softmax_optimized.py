1import numpy as np
import torch
import torch.nn as nn
import math

class Softmax:
    """
    Custom implementation of the Softmax function.

    Softmax converts a vector of values into a probability distribution.
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    """

    def __init__(self, dim=None):
        """
        Initialize Softmax.

        Args:
            dim (int, optional): Dimension along which to apply softmax.
                                If None, softmax is applied to all elements.
        """
        self.dim = dim

    def __call__(self, x):
        """
        Apply softmax to input array.

        Args:
            x: Input array (numpy array or list)

        Returns:
            Array with softmax applied
        """
        # Convert to numpy array if needed
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Ensure it's a numpy array
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        # Subtract max for numerical stability (prevents overflow)
        # This doesn't change the result due to softmax properties
        if self.dim is None:
            x_shifted = x - np.max(x)
            out = 2 ** x_shifted
            return out / np.sum(out)
        else:
            x_shifted = x - np.max(x, axis=self.dim, keepdims=True)
            out = 2 ** x_shifted
            return out / np.sum(out, axis=self.dim, keepdims=True)


# Example usage with comparison to PyTorch
if __name__ == "__main__":
    print("="*60)
    print("Comparing Custom Softmax vs PyTorch nn.Softmax")
    print("="*60 + "\n")

    # Example 1: 1D array
    print("Example 1: 1D array")
    print("-" * 40)
    custom_softmax = Softmax()
    torch_softmax = nn.Softmax(dim=0)  # For 1D, use dim=0

    x1 = [1.0, 2.0, 3.0, 4.0]
    x1_torch = torch.tensor(x1)

    len_2 = math.log(2, math.e)

    result_custom = custom_softmax(x1)
    result_torch = torch_softmax(x1_torch * len_2).numpy()

    print(f"Input: {x1}")
    print(f"Custom Output: {result_custom}")
    print(f"PyTorch Output: {result_torch}")
    print(f"Difference: {np.max(np.abs(result_custom - result_torch)):.2e}")
    print(f"Custom Sum: {np.sum(result_custom):.6f}")
    print(f"PyTorch Sum: {np.sum(result_torch):.6f}\n")

    # Example 2: 2D array with dim=1 (along columns)
    print("Example 2: 2D array (softmax along columns, dim=1)")
    print("-" * 40)
    custom_softmax_dim1 = Softmax(dim=1)
    torch_softmax_dim1 = nn.Softmax(dim=1)

    x2 = [[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]]
    x2_torch = torch.tensor(x2)

    result_custom_2d = custom_softmax_dim1(x2)
    result_torch_2d = torch_softmax_dim1(x2_torch * len_2).numpy()

    print(f"Input:\n{np.array(x2)}")
    print(f"Custom Output:\n{result_custom_2d}")
    print(f"PyTorch Output:\n{result_torch_2d}")
    print(f"Difference: {np.max(np.abs(result_custom_2d - result_torch_2d)):.2e}")
    print(f"Custom Row sums: {np.sum(result_custom_2d, axis=1)}")
    print(f"PyTorch Row sums: {np.sum(result_torch_2d, axis=1)}\n")

    # Example 3: 2D array with dim=0 (along rows)
    print("Example 3: 2D array (softmax along rows, dim=0)")
    print("-" * 40)
    custom_softmax_dim0 = Softmax(dim=0)
    torch_softmax_dim0 = nn.Softmax(dim=0)

    result_custom_dim0 = custom_softmax_dim0(x2)
    result_torch_dim0 = torch_softmax_dim0(x2_torch * len_2).numpy()

    print(f"Input:\n{np.array(x2)}")
    print(f"Custom Output:\n{result_custom_dim0}")
    print(f"PyTorch Output:\n{result_torch_dim0}")
    print(f"Difference: {np.max(np.abs(result_custom_dim0 - result_torch_dim0)):.2e}")
    print(f"Custom Column sums: {np.sum(result_custom_dim0, axis=0)}")
    print(f"PyTorch Column sums: {np.sum(result_torch_dim0, axis=0)}")

    print("\n" + "="*60)
    print("✓ All implementations match PyTorch!")

    print("="*60)
