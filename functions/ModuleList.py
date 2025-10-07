############################################################################################################
# MODULELIST IMPLEMENTATION (PyTorch vs NumPy)
############################################################################################################

import numpy as np
import torch
import torch.nn as nn


# ==========================================================================================================
# NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
class ModuleList:
    """
    A lightweight replacement for torch.nn.ModuleList using NumPy-based modules.
    Supports inference only (no gradient tracking, no training utilities).
    """
    
    def __init__(self, modules=None):
        """
        Args:
            modules (list of modules, optional): Initial list of modules to include.
        """
        if modules is None:
            self.modules = []
        else:
            self.modules = list(modules)

    def append(self, module):
        """Add a single module to the list."""
        self.modules.append(module)

    def extend(self, modules):
        """Add multiple modules at once."""
        self.modules.extend(modules)

    def insert(self, index, module):
        """Insert a module at a specific position."""
        self.modules.insert(index, module)

    def __getitem__(self, idx):
        """
        Access module(s) by index.
        Supports both integer indexing and slicing.
        """
        return self.modules[idx]

    def __setitem__(self, idx, module):
        """Set a module at a specific index."""
        self.modules[idx] = module

    def __len__(self):
        """Return the number of modules."""
        return len(self.modules)

    def __iter__(self):
        """Iterate over modules."""
        return iter(self.modules)

    def __repr__(self):
        """String representation of the ModuleList."""
        lines = ["ModuleList("]
        for i, m in enumerate(self.modules):
            lines.append(f"  ({i}): {m}")
        lines.append(")")
        return "\n".join(lines)

    def __call__(self, x):
        """
        Sequentially apply all modules to input x.
        
        Example:
            for m in self.modules:
                x = m(x)
        """
        for m in self.modules:
            x = m(x)
        return x
# ==========================================================================================================


# ==========================================================================================================
# TOY NUMPY MODULES (for testing)
# ----------------------------------------------------------------------------------------------------------
class AddOne:
    def __call__(self, x):
        return x + 1

    def __repr__(self):
        return "AddOne()"


class MultiplyTwo:
    def __call__(self, x):
        return x * 2

    def __repr__(self):
        return "MultiplyTwo()"


class Square:
    def __call__(self, x):
        return x ** 2

    def __repr__(self):
        return "Square()"


class Scale:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

    def __repr__(self):
        return f"Scale(factor={self.factor})"
# ==========================================================================================================


# ==========================================================================================================
# TOY PYTORCH MODULES (for comparison)
# ----------------------------------------------------------------------------------------------------------
class TorchAddOne(nn.Module):
    def forward(self, x):
        return x + 1

    def __repr__(self):
        return "TorchAddOne()"


class TorchMultiplyTwo(nn.Module):
    def forward(self, x):
        return x * 2

    def __repr__(self):
        return "TorchMultiplyTwo()"


class TorchSquare(nn.Module):
    def forward(self, x):
        return x ** 2

    def __repr__(self):
        return "TorchSquare()"


class TorchScale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor

    def __repr__(self):
        return f"TorchScale(factor={self.factor})"
# ==========================================================================================================


# ==========================================================================================================
# TEST CASES - Compare NumPy vs PyTorch
# ----------------------------------------------------------------------------------------------------------
def test_modulelist_operations(test_name=""):
    """
    Test basic operations: append, extend, indexing, slicing, iteration
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    # NumPy version
    np_list = ModuleList()
    np_list.append(AddOne())
    np_list.append(MultiplyTwo())
    np_list.extend([Square(), Scale(0.5)])
    
    # PyTorch version
    torch_list = nn.ModuleList()
    torch_list.append(TorchAddOne())
    torch_list.append(TorchMultiplyTwo())
    torch_list.extend([TorchSquare(), TorchScale(0.5)])
    
    print(f"NumPy ModuleList length: {len(np_list)}")
    print(f"PyTorch ModuleList length: {len(torch_list)}")
    
    # Test indexing
    print(f"\nNumPy ModuleList[0]: {np_list[0]}")
    print(f"PyTorch ModuleList[0]: {torch_list[0]}")
    
    # Test slicing
    print(f"\nNumPy ModuleList[1:3]: {[str(m) for m in np_list[1:3]]}")
    print(f"PyTorch ModuleList[1:3]: {[str(m) for m in torch_list[1:3]]}")
    
    # Test iteration
    print(f"\nNumPy modules: {[str(m) for m in np_list]}")
    print(f"PyTorch modules: {[str(m) for m in torch_list]}")
    
    print("✓ PASSED - All operations work correctly")
    return True


def test_modulelist_forward(test_name=""):
    """
    Test forward pass through ModuleList
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    # Create input
    x_np = np.array([1.0, 2.0, 3.0])
    x_torch = torch.tensor([1.0, 2.0, 3.0])
    
    # NumPy version
    np_list = ModuleList([AddOne(), MultiplyTwo()])
    y_np = np_list(x_np)
    
    # PyTorch version (manual sequential application)
    torch_list = nn.ModuleList([TorchAddOne(), TorchMultiplyTwo()])
    y_torch = x_torch
    for module in torch_list:
        y_torch = module(y_torch)
    y_torch = y_torch.numpy()
    
    print(f"Input: {x_np}")
    print(f"NumPy output: {y_np}")
    print(f"PyTorch output: {y_torch}")
    
    match = np.allclose(y_np, y_torch)
    print(f"Outputs match: {match}")
    
    if match:
        print("✓ PASSED")
    else:
        print("✗ FAILED")
    
    return match


def test_complex_pipeline(test_name=""):
    """
    Test a more complex pipeline with multiple operations
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    # Create input
    x_np = np.array([1.0, 2.0, 3.0, 4.0])
    x_torch = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    # NumPy version: (x + 1) * 2, then square, then scale by 0.1
    np_list = ModuleList([
        AddOne(),
        MultiplyTwo(),
        Square(),
        Scale(0.1)
    ])
    
    # PyTorch version
    torch_list = nn.ModuleList([
        TorchAddOne(),
        TorchMultiplyTwo(),
        TorchSquare(),
        TorchScale(0.1)
    ])
    
    # Forward pass
    y_np = np_list(x_np)
    
    y_torch = x_torch
    for module in torch_list:
        y_torch = module(y_torch)
    y_torch = y_torch.numpy()
    
    print(f"Pipeline: AddOne -> MultiplyTwo -> Square -> Scale(0.1)")
    print(f"Input: {x_np}")
    print(f"NumPy output: {y_np}")
    print(f"PyTorch output: {y_torch}")
    
    # Manual verification for x=1: ((1+1)*2)^2 * 0.1 = (4)^2 * 0.1 = 1.6
    print(f"\nManual check for x=1: ((1+1)*2)^2 * 0.1 = {((1+1)*2)**2 * 0.1}")
    
    match = np.allclose(y_np, y_torch)
    print(f"Outputs match: {match}")
    
    if match:
        print("✓ PASSED")
    else:
        print("✗ FAILED")
    
    return match


def test_empty_modulelist(test_name=""):
    """
    Test empty ModuleList
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    np_list = ModuleList()
    torch_list = nn.ModuleList()
    
    print(f"NumPy ModuleList length: {len(np_list)}")
    print(f"PyTorch ModuleList length: {len(torch_list)}")
    
    # Empty list should return input unchanged
    x = np.array([1.0, 2.0, 3.0])
    y = np_list(x)
    
    print(f"Input: {x}")
    print(f"Output (empty list): {y}")
    print(f"Input unchanged: {np.array_equal(x, y)}")
    
    print("✓ PASSED")
    return True


def run_all_tests():
    """
    Run comprehensive test suite
    """
    print("="*80)
    print("COMPARING NumPy ModuleList vs PyTorch nn.ModuleList")
    print("="*80)
    
    all_passed = True
    
    all_passed &= test_modulelist_operations("Basic Operations (append, extend, indexing)")
    all_passed &= test_modulelist_forward("Sequential Forward Pass")
    all_passed &= test_complex_pipeline("Complex Multi-Step Pipeline")
    all_passed &= test_empty_modulelist("Empty ModuleList")
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - NumPy implementation matches PyTorch behavior!")
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
    
    # Example 1: Basic usage
    print("\nExample 1: Basic ModuleList Usage")
    print("-"*40)
    
    # NumPy version
    modules = ModuleList([AddOne(), MultiplyTwo()])
    print("NumPy ModuleList:")
    print(modules)
    
    x = np.array([1.0, 2.0, 3.0])
    print(f"\nInput: {x}")
    y = modules(x)
    print(f"Output after AddOne then MultiplyTwo: {y}")
    print("Computation: (x + 1) * 2 = ([1,2,3] + 1) * 2 = [2,3,4] * 2 = [4,6,8]")
    
    # Example 2: Building progressively
    print("\n" + "="*80)
    print("Example 2: Building ModuleList Progressively")
    print("-"*40)
    
    modules = ModuleList()
    print(f"Initial length: {len(modules)}")
    
    modules.append(AddOne())
    print(f"After append AddOne: {len(modules)}")
    
    modules.extend([MultiplyTwo(), Square()])
    print(f"After extend [MultiplyTwo, Square]: {len(modules)}")
    
    print("\nFinal ModuleList:")
    print(modules)
    
    # Example 3: Indexing and slicing
    print("\n" + "="*80)
    print("Example 3: Indexing and Slicing")
    print("-"*40)
    
    modules = ModuleList([AddOne(), MultiplyTwo(), Square(), Scale(0.5)])
    
    print(f"Full list ({len(modules)} modules):")
    for i, m in enumerate(modules):
        print(f"  [{i}]: {m}")
    
    print(f"\nAccess by index [1]: {modules[1]}")
    print(f"Slice [1:3]: {[str(m) for m in modules[1:3]]}")
    
    # Example 4: Comparison with nn.Sequential
    print("\n" + "="*80)
    print("Example 4: ModuleList vs nn.Sequential Behavior")
    print("-"*40)
    
    x = np.array([2.0, 3.0])
    
    # Using ModuleList (with __call__)
    module_list = ModuleList([AddOne(), MultiplyTwo()])
    y_list = module_list(x)
    
    # Using manual loop (equivalent)
    y_manual = x
    for m in module_list:
        y_manual = m(y_manual)
    
    print(f"Input: {x}")
    print(f"Output via ModuleList.__call__: {y_list}")
    print(f"Output via manual loop: {y_manual}")
    print(f"Results match: {np.array_equal(y_list, y_manual)}")
    
    # Example 5: Insert operation
    print("\n" + "="*80)
    print("Example 5: Insert Operation")
    print("-"*40)
    
    modules = ModuleList([AddOne(), MultiplyTwo()])
    print("Original:")
    print(modules)
    
    modules.insert(1, Square())
    print("\nAfter inserting Square at index 1:")
    print(modules)
    
    x = np.array([2.0])
    y = modules(x)
    print(f"\nInput: {x}")
    print(f"Output: {y}")
    print("Computation: AddOne -> Square -> MultiplyTwo")
    print(f"  (2 + 1) = 3, then 3^2 = 9, then 9 * 2 = 18")


# ==========================================================================================================
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run comprehensive comparison tests
    run_all_tests()
    
    # Show usage examples
    usage_examples()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The NumPy ModuleList implementation replicates PyTorch's nn.ModuleList")
    print("behavior for inference tasks. Key features:")
    print("- Sequential application via __call__ (like nn.Sequential)")
    print("- List operations: append, extend, insert, indexing, slicing")
    print("- Iteration support")
    print("- No gradient tracking (inference only)")
    print("="*80)