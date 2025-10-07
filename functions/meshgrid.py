############################################################################################################
# MESHGRID IMPLEMENTATION (NumPy vs Manual)
############################################################################################################

import numpy as np


# ==========================================================================================================
# MANUAL IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------
def meshgrid(x, y):
    """
    Manually creates 2D coordinate matrices from 1D coordinate vectors 
    (equivalent to np.meshgrid(x, y, indexing='xy')).

    Args:
        x (np.ndarray): 1D array of length Nx (e.g., width coordinates).
        y (np.ndarray): 1D array of length Ny (e.g., height coordinates).

    Returns:
        tuple (X, Y): 
            X (np.ndarray): Shape (Ny, Nx), where x values are repeated vertically.
            Y (np.ndarray): Shape (Ny, Nx), where y values are repeated horizontally.
    """
    Ny = len(y)
    Nx = len(x)

    # 1. Create the X matrix: Repeat the x-vector (row) Ny times (downward)
    X = np.zeros((Ny, Nx), dtype=x.dtype)
    for i in range(Ny):
        X[i, :] = x
    
    # 2. Create the Y matrix: Repeat the y-vector (column) Nx times (across)
    Y = np.zeros((Ny, Nx), dtype=y.dtype)
    for j in range(Nx):
        Y[:, j] = y
        
    return X, Y
# ==========================================================================================================


# ==========================================================================================================
# TEST CASES - Compare Manual vs NumPy
# ----------------------------------------------------------------------------------------------------------
def test_meshgrid(x, y, test_name=""):
    """
    Compare manual meshgrid against NumPy's np.meshgrid
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    print(f"x shape: {x.shape}, values: {x}")
    print(f"y shape: {y.shape}, values: {y}")
    print(f"{'-'*80}")
    
    # Manual implementation
    X_manual, Y_manual = meshgrid(x, y)
    
    # NumPy implementation
    X_numpy, Y_numpy = np.meshgrid(x, y, indexing='xy')
    
    # Compare results
    X_match = np.allclose(X_manual, X_numpy)
    Y_match = np.allclose(Y_manual, Y_numpy)
    shapes_match = (X_manual.shape == X_numpy.shape) and (Y_manual.shape == Y_numpy.shape)
    
    print(f"Manual X shape: {X_manual.shape}")
    print(f"NumPy X shape:  {X_numpy.shape}")
    print(f"Manual Y shape: {Y_manual.shape}")
    print(f"NumPy Y shape:  {Y_numpy.shape}")
    print(f"Shapes match: {shapes_match}")
    print(f"X arrays match: {X_match}")
    print(f"Y arrays match: {Y_match}")
    
    if X_manual.size <= 50:  # Only print small arrays
        print(f"\nManual X:\n{X_manual}")
        print(f"NumPy X:\n{X_numpy}")
        print(f"\nManual Y:\n{Y_manual}")
        print(f"NumPy Y:\n{Y_numpy}")
    else:
        print(f"\nX sample (first 3x3):\n{X_manual[:3, :3]}")
        print(f"Y sample (first 3x3):\n{Y_manual[:3, :3]}")
    
    all_match = X_match and Y_match and shapes_match
    if all_match:
        print("✓ PASSED")
    else:
        print("✗ FAILED")
    
    return all_match


def run_all_tests():
    """
    Run comprehensive test suite
    """
    print("="*80)
    print("COMPARING Manual meshgrid vs NumPy np.meshgrid")
    print("="*80)
    
    all_passed = True
    
    # Test 1: Small integer arrays
    all_passed &= test_meshgrid(
        x=np.array([0, 1, 2]),
        y=np.array([10, 20]),
        test_name="Small integers: x=[0,1,2], y=[10,20]"
    )
    
    # Test 2: Equal size arrays
    all_passed &= test_meshgrid(
        x=np.array([1, 2, 3, 4]),
        y=np.array([5, 6, 7, 8]),
        test_name="Equal sizes: 4x4 grid"
    )
    
    # Test 3: Float arrays
    all_passed &= test_meshgrid(
        x=np.array([0.0, 0.5, 1.0, 1.5]),
        y=np.array([0.0, 1.0, 2.0]),
        test_name="Float coordinates"
    )
    
    # Test 4: Single element arrays
    all_passed &= test_meshgrid(
        x=np.array([5]),
        y=np.array([10]),
        test_name="Single point: x=[5], y=[10]"
    )
    
    # Test 5: Large x, small y
    all_passed &= test_meshgrid(
        x=np.arange(0, 10),
        y=np.array([0, 1]),
        test_name="Wide grid: 10x2"
    )
    
    # Test 6: Small x, large y
    all_passed &= test_meshgrid(
        x=np.array([0, 1]),
        y=np.arange(0, 10),
        test_name="Tall grid: 2x10"
    )
    
    # Test 7: Negative values
    all_passed &= test_meshgrid(
        x=np.array([-2, -1, 0, 1, 2]),
        y=np.array([-1, 0, 1]),
        test_name="Negative coordinates"
    )
    
    # Test 8: Large grid
    all_passed &= test_meshgrid(
        x=np.arange(0, 20),
        y=np.arange(0, 15),
        test_name="Large grid: 20x15"
    )
    
    # Test 9: Non-uniform spacing
    all_passed &= test_meshgrid(
        x=np.array([0, 1, 3, 6, 10]),
        y=np.array([0, 2, 5]),
        test_name="Non-uniform spacing"
    )
    
    # Test 10: Float32 dtype
    all_passed &= test_meshgrid(
        x=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        y=np.array([4.0, 5.0], dtype=np.float32),
        test_name="Float32 dtype"
    )
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Manual implementation matches NumPy!")
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
    print("\nExample 1: Basic 3x2 Grid")
    print("-"*40)
    
    x = np.array([0, 1, 2])
    y = np.array([10, 20])
    
    print(f"x coordinates: {x}")
    print(f"y coordinates: {y}")
    
    X_manual, Y_manual = meshgrid(x, y)
    X_numpy, Y_numpy = np.meshgrid(x, y, indexing='xy')
    
    print("\nManual meshgrid:")
    print(f"X:\n{X_manual}")
    print(f"Y:\n{Y_manual}")
    
    print("\nNumPy meshgrid:")
    print(f"X:\n{X_numpy}")
    print(f"Y:\n{Y_numpy}")
    
    print(f"\nMatch: {np.allclose(X_manual, X_numpy) and np.allclose(Y_manual, Y_numpy)}")
    
    # Example 2: Visualization explanation
    print("\n" + "="*80)
    print("Example 2: Understanding Meshgrid")
    print("-"*40)
    
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 10, 20])
    
    X, Y = meshgrid(x, y)
    
    print(f"x: {x} (width coordinates)")
    print(f"y: {y} (height coordinates)")
    print(f"\nResulting grid points (x, y):")
    print(f"Shape: {X.shape} = (Ny={len(y)}, Nx={len(x)})")
    print("\nX (x-coordinates at each grid point):")
    print(X)
    print("\nY (y-coordinates at each grid point):")
    print(Y)
    print("\nGrid points as (x, y) pairs:")
    for i in range(len(y)):
        for j in range(len(x)):
            print(f"  ({X[i,j]:4.0f}, {Y[i,j]:4.0f})", end="")
        print()
    
    # Example 3: Image coordinate system
    print("\n" + "="*80)
    print("Example 3: Image Pixel Coordinates")
    print("-"*40)
    
    width, height = 5, 3
    x = np.arange(width)
    y = np.arange(height)
    
    X, Y = meshgrid(x, y)
    
    print(f"Image size: {width}x{height} (width x height)")
    print(f"\nPixel x-coordinates:\n{X}")
    print(f"\nPixel y-coordinates:\n{Y}")
    print(f"\nTotal pixels: {X.size}")
    
    # Example 4: Function evaluation over grid
    print("\n" + "="*80)
    print("Example 4: Evaluate Function over Grid")
    print("-"*40)
    
    x = np.linspace(-2, 2, 5)
    y = np.linspace(-1, 1, 3)
    
    X, Y = meshgrid(x, y)
    
    # Evaluate f(x,y) = x^2 + y^2 (distance from origin)
    Z = X**2 + Y**2
    
    print(f"x range: [{x[0]:.1f}, {x[-1]:.1f}]")
    print(f"y range: [{y[0]:.1f}, {y[-1]:.1f}]")
    print(f"\nX grid:\n{X}")
    print(f"\nY grid:\n{Y}")
    print(f"\nZ = X² + Y² (distance squared):\n{Z}")
    
    # Example 5: Comparison with 'ij' indexing
    print("\n" + "="*80)
    print("Example 5: 'xy' vs 'ij' Indexing")
    print("-"*40)
    
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    
    X_xy, Y_xy = np.meshgrid(x, y, indexing='xy')
    X_ij, Y_ij = np.meshgrid(x, y, indexing='ij')
    X_manual, Y_manual = meshgrid(x, y)
    
    print("Our implementation uses 'xy' indexing (Cartesian coordinates)")
    print(f"\nWith indexing='xy' (our implementation):")
    print(f"X shape: {X_manual.shape} (Ny, Nx) = ({len(y)}, {len(x)})")
    print(f"X:\n{X_manual}")
    print(f"Y:\n{Y_manual}")
    
    print(f"\nWith indexing='ij' (matrix indexing):")
    print(f"X shape: {X_ij.shape} (Nx, Ny) = ({len(x)}, {len(y)})")
    print(f"X:\n{X_ij}")
    print(f"Y:\n{Y_ij}")
    
    print(f"\nOur implementation matches 'xy': {np.allclose(X_manual, X_xy)}")


# ==========================================================================================================
# VISUALIZATION HELPER
# ----------------------------------------------------------------------------------------------------------
def visualize_grid(x, y):
    """
    Create a simple text visualization of the grid
    """
    print("\n" + "="*80)
    print("GRID VISUALIZATION")
    print("="*80)
    
    X, Y = meshgrid(x, y)
    
    print(f"Grid: {len(x)} x {len(y)} (width x height)")
    print("\nCoordinate pairs (x, y):\n")
    
    for i in range(len(y)-1, -1, -1):  # Print from top to bottom
        print(f"y={y[i]:5.1f} | ", end="")
        for j in range(len(x)):
            print(f"({X[i,j]:4.1f},{Y[i,j]:4.1f}) ", end="")
        print()
    
    print(" " * 9 + "-" * (len(x) * 13))
    print(" " * 9, end="")
    for xi in x:
        print(f"   x={xi:4.1f}   ", end="")
    print()


# ==========================================================================================================
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run comprehensive comparison tests
    run_all_tests()
    
    # Show usage examples
    usage_examples()
    
    # Visualize a sample grid
    visualize_grid(
        x=np.array([0, 1, 2, 3]),
        y=np.array([0, 1, 2])
    )
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The manual meshgrid implementation perfectly replicates")
    print("NumPy's np.meshgrid(x, y, indexing='xy') behavior.")
    print("\nKey Points:")
    print("- X matrix: x values repeated vertically (shape: Ny × Nx)")
    print("- Y matrix: y values repeated horizontally (shape: Ny × Nx)")
    print("- Uses 'xy' (Cartesian) indexing, not 'ij' (matrix) indexing")
    print("="*80)