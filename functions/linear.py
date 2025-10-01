import numpy as np

class ExplicitLinear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights and bias like PyTorch: normal distribution
        self.weight = np.random.randn(out_features, in_features)  # [out_features, in_features]
        if self.use_bias:
            self.bias = np.random.randn(out_features)  # [out_features]
        else:
            self.bias = None
    
    def forward(self, input_data):
        """
        Args:
            input_data: numpy array of shape [batch_size, in_features] or [..., in_features]
        Returns:
            output: numpy array of shape [batch_size, out_features] or [..., out_features]
        """
        # Handle multi-dimensional inputs
        original_shape = input_data.shape
        input_data = input_data.reshape(-1, self.in_features)
        
        batch_size = input_data.shape[0]
        output = np.zeros((batch_size, self.out_features))
        
        for i in range(batch_size):
            for j in range(self.out_features):
                sum_val = 0.0
                for k in range(self.in_features):
                    sum_val += input_data[i, k] * self.weight[j, k]
                if self.use_bias:
                    sum_val += self.bias[j]
                output[i, j] = sum_val
        
        # Reshape back to original dimensions
        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = output.reshape(new_shape)
        return output
    
    def __call__(self, input_data):
        return self.forward(input_data)

