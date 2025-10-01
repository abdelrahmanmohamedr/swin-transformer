import math

class AdaptiveAvgPool1d:
    def __init__(self, output_size):
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.output_size = output_size

    def _pool_1d(self, data):
        input_size = len(data)
        result = []
        for i in range(self.output_size):
            start = int(math.floor(i * input_size / self.output_size))
            end = int(math.ceil((i + 1) * input_size / self.output_size))
            segment = data[start:end]
            avg = sum(segment) / len(segment) if segment else 0.0
            result.append(avg)
        return result

    def __call__(self, data):
        # 1D input: [L]
        if isinstance(data[0], (int, float)):
            return self._pool_1d(data)

        # 2D input: [C][L]
        elif isinstance(data[0], list) and isinstance(data[0][0], (int, float)):
            return [self._pool_1d(channel) for channel in data]

        # 3D input: [N][C][L]
        elif isinstance(data[0], list) and isinstance(data[0][0], list) and isinstance(data[0][0][0], (int, float)):
            return [
                [self._pool_1d(channel) for channel in sample]
                for sample in data
            ]

        else:
            raise ValueError("Unsupported input format")