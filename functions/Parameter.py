class Parameter:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        return f"ManualParameter(shape={self.shape()}, requires_grad={self.requires_grad})"

    def shape(self):
        # recursively compute nested list shape
        def _shape(x):
            if isinstance(x, list):
                return [len(x)] + _shape(x[0])
            else:
                return []
        return tuple(_shape(self.data))

import torch
import torch.nn as nn

# Example sizes
Wh, Ww, num_heads = 7,7, 2
num_patches, embed_dim = 4, 3

# Relative position bias table (API)
bias_table_api = nn.Parameter(torch.zeros((2*Wh-1)*(2*Ww-1), num_heads))

# Absolute position embedding (API)
abs_pos_api = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

print("API Relative Bias Table Shape:", bias_table_api.shape)
print("API Absolute Pos Embed Shape:", abs_pos_api.shape)

# Relative Position Bias Table (Manual)
num_positions = (2*Wh - 1) * (2*Ww - 1)
bias_table_manual_data = [[0.0 for _ in range(num_heads)] for _ in range(num_positions)]
bias_table_manual = ManualParameter(bias_table_manual_data)

# Absolute Position Embedding (Manual)
abs_pos_manual_data = [[[0.0 for _ in range(embed_dim)] for _ in range(num_patches)]]
abs_pos_manual = ManualParameter(abs_pos_manual_data)

print("Manual Relative Bias Table Shape:", bias_table_manual.shape())
print("Manual Absolute Pos Embed Shape:", abs_pos_manual.shape())
