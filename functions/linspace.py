def linspace_list(start, end, steps):
    if steps == 1:
        return [start]
    step_size = (end - start) / (steps - 1)
    return [start + i * step_size for i in range(steps)]