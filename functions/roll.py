def roll_nd_list(arr, shifts, dims):
    # Ensure shifts and dims are tuples for consistency
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)

    # Helper function to determine the shape of the nested list
    def get_shape(a):
        shape = []
        while isinstance(a, list):
            shape.append(len(a))
            a = a[0] if a else []  # Move deeper into the nested list
        return shape

    shape = get_shape(arr)

    # Helper function to create an empty nested list with the same shape
    def make_empty(shape):
        if not shape:
            return None  # Base case: scalar value
        return [make_empty(shape[1:]) for _ in range(shape[0])]

    result = make_empty(shape)  # Initialize the result container

    # Recursive function to traverse the original list and place values in their new rolled positions
    def recurse(idx, subarr):
        if not isinstance(subarr, list):  # Base case: reached a scalar value
            new_idx = list(idx)  # Copy the current index path
            for s, d in zip(shifts, dims):
                # Apply the shift to the specified dimension with wrap-around using modulo
                new_idx[d] = (new_idx[d] + s) % shape[d]

            # Navigate to the correct location in the result list and assign the value
            ref = result
            for i in new_idx[:-1]:
                ref = ref[i]
            ref[new_idx[-1]] = subarr
        else:
            # Recursive case: iterate through the current level and go deeper
            for i, val in enumerate(subarr):
                recurse(idx + [i], val)

    # Start recursion from the root of the list
    recurse([], arr)
    return result