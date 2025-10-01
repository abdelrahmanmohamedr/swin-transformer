import numpy as np

def manual_concatenate(arrays, axis=0):
    """
    Manually concatenates a sequence of NumPy arrays along a specified axis 
    (equivalent to np.concatenate(arrays, axis)).
    
    Args:
        arrays (list of np.ndarray): The arrays to concatenate. Must have same shape
                                     in all dimensions except the concatenation axis.
        axis (int): The axis along which to concatenate. Default is 0.
        
    Returns:
        np.ndarray: The concatenated array.
    """
    if not arrays:
        raise ValueError("Input 'arrays' list cannot be empty.")
        
    # Get the number of arrays and their shape/rank
    N = len(arrays)
    rank = arrays[0].ndim
    
    # Handle negative axis index
    if axis < 0:
        axis = rank + axis
        
    if not (0 <= axis < rank):
        raise ValueError(f"Axis {axis} is out of bounds for array with rank {rank}.")

    # 1. Determine the output shape
    output_shape = list(arrays[0].shape)
    
    # Sum the sizes of the concatenation axis
    concat_size = sum(arr.shape[axis] for arr in arrays)
    output_shape[axis] = concat_size
    
    # 2. Initialize the output array
    output = np.empty(output_shape, dtype=arrays[0].dtype)
    
    # 3. Copy data piece by piece
    current_index = 0
    
    # Create the slicing object for the copy operation
    # E.g., if axis=0, the slice is [start:end, :, :, ...]
    # If axis=1, the slice is [:, start:end, :, ...]
    
    for arr in arrays:
        # Determine the slice range for the current array along the concatenation axis
        arr_size = arr.shape[axis]
        
        # Create a list of slices for all dimensions
        slices = [slice(None)] * rank
        slices[axis] = slice(current_index, current_index + arr_size)
        
        # Convert list of slices to a tuple for numpy indexing
        slice_tuple = tuple(slices)
        
        # Copy the current array into the corresponding segment of the output array
        output[slice_tuple] = arr
        
        current_index += arr_size
        
    return output

# Example usage (for testing):
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# c = manual_concatenate([a, b], axis=0) # Concatenate rows
# d = manual_concatenate([a, b.T], axis=1) # Concatenate columns
# print("Concatenated Axis 0:\n", c)
# print("Concatenated Axis 1:\n", d)
