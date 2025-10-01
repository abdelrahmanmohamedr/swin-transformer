import numpy as np
# Import manually implemented primitives
from gelu import GELU
from fc_layer import fc_layer_explicit

class ManualMlp:
    """
    Manual, NumPy-based implementation of the Swin Transformer MLP block (Feed-Forward Network).
    This block performs the expansion, activation, and contraction steps.
    """
    def __init__(self, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray):
        """
        Args:
            W1, B1: Weight and bias for the expansion layer (FC1).
            W2, B2: Weight and bias for the contraction layer (FC2).
        """
        # FC1 (Expansion layer parameters)
        self.fc1_W = W1  # [hidden_features, in_features]
        self.fc1_B = B1  # [hidden_features]
        
        # FC2 (Contraction layer parameters)
        self.fc2_W = W2  # [out_features, hidden_features]
        self.fc2_B = B2  # [out_features]
        
        # Activation function
        self.act = GELU()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the MLP. Input x shape: [Total_Tokens, C]
        """
        # 1. FC1 (Expansion: C -> 4C)
        x = fc_layer_explicit(x, self.fc1_W, self.fc1_B)

        # 2. GELU Activation
        x = self.act(x)

        # 3. FC2 (Contraction: 4C -> C) - Dropout is implicitly skipped for inference
        x = fc_layer_explicit(x, self.fc2_W, self.fc2_B)

        return x
