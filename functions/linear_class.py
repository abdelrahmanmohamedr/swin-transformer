import numpy as np

class ExplicitLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias like PyTorch: normal distribution
        self.weight = np.random.randn(out_features, in_features)  # [out_features, in_features]
        self.bias = np.random.randn(out_features)                 # [out_features]

    def forward(self, input_data):
        """
        Args:
            input_data: [batch_size, in_features]
        Returns:
            output: [batch_size, out_features]
        """
        batch_size, feature_size = input_data.shape
        if feature_size != self.in_features:
            raise ValueError(f"Expected input with {self.in_features} features, got {feature_size}")

        output = np.zeros((batch_size, self.out_features))
        for i in range(batch_size):
            for j in range(self.out_features):
                sum_val = 0.0
                for k in range(self.in_features):
                    sum_val += input_data[i, k] * self.weight[j, k]
                sum_val += self.bias[j]
                output[i, j] = sum_val
        return output

    def __call__(self, input_data):
        return self.forward(input_data)