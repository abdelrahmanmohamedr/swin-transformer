############################################################################################################
# NumPy-based ModuleList (for Inference Only)
# ----------------------------------------------------------------------------------------------------------
# This script defines a lightweight replacement for torch.nn.ModuleList, written in NumPy style and intended
# for inference tasks only (no gradient tracking, no training utilities). It supports:
#   - append(), extend(), insert()
#   - indexing, slicing, iteration
#   - sequential application of modules via __call__ (like nn.Sequential)
############################################################################################################
import numpy as np
class ModuleList:
    def __init__(self, modules=None):
        """
        A lightweight replacement for torch.nn.ModuleList
        using NumPy-based modules.

        Args:
            modules (list of modules, optional): 
                Initial list of modules to include.
        """
        if modules is None:
            self.modules = []
        else:
            self.modules = list(modules)

    def append(self, module):
        """
        Add a single module to the list.
        """
        self.modules.append(module)

    def extend(self, modules):
        """
        Add multiple modules at once.
        """
        self.modules.extend(modules)

    def __getitem__(self, idx):
        """
        Access module(s) by index.
        Supports both integer indexing and slicing.
        """
        return self.modules[idx]

    def __len__(self):
        """
        Return the number of modules.
        """
        return len(self.modules)

    def __iter__(self):
        """
        Iterate over modules.
        """
        return iter(self.modules)

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
# TOY MODULE DEFINITIONS (for demonstration)
# ----------------------------------------------------------------------------------------------------------
# These act as "layers" that process input NumPy arrays.
# Each has __call__ for execution and __repr__ for debugging/printing.
############################################################################################################
"""
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
"""
# ==========================================================================================================
# SIMPLE USAGE EXAMPLE
# ----------------------------------------------------------------------------------------------------------
# 1. Create a ModuleList containing AddOne and MultiplyTwo
# 2. Print its contents
# 3. Apply it to a NumPy array and show input/output
############################################################################################################
"""
modules = ModuleList([AddOne(), MultiplyTwo()])

print("Modules inside ModuleList:")
print(modules)

# Input array
x = np.array([1, 2, 3], dtype=float)
print("\nInput:", x)

# Run inference (AddOne first, then MultiplyTwo)
y = modules(x)
print("Output:", y)
"""
# ==========================================================================================================
# EXPECTED OUTPUT
# ----------------------------------------------------------------------------------------------------------
# Modules inside ModuleList:
# ModuleList(
#   AddOne()
#   MultiplyTwo()
# )
#
# Input: [1. 2. 3.]
# Output: [ 4. 6. 8.]
############################################################################################################      
