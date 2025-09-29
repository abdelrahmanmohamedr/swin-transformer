import math

def adaptive_avg_pool_1d_list(data, output_size):
    input_size = len(data)
    if output_size <= 0 or input_size == 0:
        return []

    result = []
    for i in range(output_size):
        start = math.floor(i * input_size / output_size)
        end = math.ceil((i + 1) * input_size / output_size)
        segment = data[start:end]
        avg = sum(segment) / len(segment)
        result.append(avg)
    return result

def adaptive_avg_pool(data, output_size):
    # 1D input: [L]
    if isinstance(data[0], (int, float)):
        return adaptive_avg_pool_1d_list(data, output_size)

    # 2D input: [C][L]
    elif isinstance(data[0][0], (int, float)):
        return [adaptive_avg_pool_1d_list(channel, output_size) for channel in data]

    # 3D input: [N][C][L]
    elif isinstance(data[0][0][0], (int, float)):
        return [
            [adaptive_avg_pool_1d_list(channel, output_size) for channel in sample]
            for sample in data
        ]

    else:
        raise ValueError("Unsupported input format")