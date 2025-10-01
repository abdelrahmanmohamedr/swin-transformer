import numpy as np

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
    #    The inner dimension is the x-dimension (Nx).
    X = np.zeros((Ny, Nx), dtype=x.dtype)
    for i in range(Ny):
        # Assign the entire x-vector to the current row i
        X[i, :] = x
    
    # 2. Create the Y matrix: Repeat the y-vector (column) Nx times (across)
    #    The inner dimension is the x-dimension (Nx).
    Y = np.zeros((Ny, Nx), dtype=y.dtype)
    for j in range(Nx):
        # Assign the entire y-vector to the current column j
        # y[:, None] creates a column vector [Ny, 1] which broadcasts easily
        Y[:, j] = y
        
    return X, Y

# Example usage (for testing):
# x_coords = np.array([0, 1, 2])
# y_coords = np.array([10, 20])
# X_grid, Y_grid = manual_meshgrid(x_coords, y_coords)
# print("X_grid:\n", X_grid)
# print("Y_grid:\n", Y_grid)
