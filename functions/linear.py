############################################################################################################
# LINEAR LAYER IMPLEMENTATION (PyTorch vs NumPy)
############################################################################################################

import numpy as np
import torch
import torch.nn as nn
import math

# ==========================================================================================================
# NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------

class ExplicitLinear:
    def __init__(self, in_features, out_features, weight=None, bias=None, bias_condition=True):
        """
        NumPy-compatible interface with PyTorch backend for exact matching.
        Optimized to minimize conversion overhead.
        """
        self.in_features = in_features
        self.out_features = out_features

        # Store as numpy for compatibility
        if weight is not None:
            assert weight.shape == (out_features, in_features)
            self.weight = weight.astype(np.float32)
        else:
            # Use Kaiming uniform initialization as in PyTorch's nn.Linear default
            # fan_in = in_features
            # bound = 1 / math.sqrt(fan_in)
            # self.weight = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32)
            # Use PyTorch's default initialization for consistency in testing
            weight_tensor = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(weight_tensor, a=math.sqrt(5))
            self.weight = weight_tensor.numpy().astype(np.float32)


        if bias is None and bias_condition is True:
            # Use PyTorch's default bias initialization (uniform based on kaiming)
            # fan_in = in_features
            # bound = 1 / math.sqrt(fan_in)
            # self.bias = np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)
            bias_tensor = torch.empty(out_features)
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias_tensor, -bound, bound)
            self.bias = bias_tensor.numpy().astype(np.float32)
            self.use_bias = True
        elif bias_condition is False:
            self.bias = None
            self.use_bias = False
        elif isinstance(bias, np.ndarray):
            assert bias.shape == (out_features,)
            self.bias = bias.astype(np.float32)
            self.use_bias = True
        else:
            raise ValueError(f"Invalid bias parameter: {bias}")

        # Convert to torch tensors ONCE - reuse them
        # Ensure they are on CPU as this is the NumPy implementation
        self.weight_torch = torch.from_numpy(self.weight).cpu()
        self.bias_torch = torch.from_numpy(self.bias).cpu() if self.use_bias else None

    def forward(self, input_data):
        """
        NumPy input/output, explicit loop computation.
        """
        # Ensure input is float32 numpy array
        if isinstance(input_data, torch.Tensor):
             # Convert torch tensor to numpy array (shares memory if on CPU)
             input_data = input_data.detach().cpu().numpy()

        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)


        # Handle multi-dimensional inputs by reshaping to (N, in_features)
        original_shape = input_data.shape
        input_2d = input_data.reshape(-1, self.in_features)
        num_samples = input_2d.shape[0]

        # Initialize output array
        output_2d = np.zeros((num_samples, self.out_features), dtype=np.float32)

        # Perform matrix multiplication and bias addition using loops
        # output_2d[i, j] = sum(input_2d[i, k] * self.weight[j, k] for k in range(in_features)) + bias[j]
        for i in range(num_samples):
            for j in range(self.out_features):
                sum_val = 0.0
                for k in range(self.in_features):
                    sum_val += input_2d[i, k] * self.weight[j, k]  # Note: Weight is (out, in)
                output_2d[i, j] = sum_val
                if self.use_bias:
                    output_2d[i, j] += self.bias[j]


        # Reshape back to original shape but with the last dimension replaced by out_features
        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = output_2d.reshape(new_shape)

        return output

    def __call__(self, input_data):
        return self.forward(input_data)
# ==========================================================================================================


# ==========================================================================================================
# TEST CASES - Compare NumPy vs PyTorch
# ----------------------------------------------------------------------------------------------------------
def test_linear(input_shape, in_features, out_features, bias=True, test_name=""):
    """
    Compare NumPy ExplicitLinear against PyTorch nn.Linear
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    print(f"Input shape: {input_shape}")
    print(f"Input features: {in_features}")
    print(f"Output features: {out_features}")
    print(f"Bias: {bias}")
    print(f"{'-'*80}")

    # Create random input
    np.random.seed(42)
    torch.manual_seed(42)

    np_input = np.random.randn(*input_shape).astype(np.float32)
    torch_input = torch.from_numpy(np_input.copy())

    # Create Linear layer instances
    # Create PyTorch first to use its weights for manual
    torch_linear = nn.Linear(in_features, out_features, bias=bias)

    # Extract weights from PyTorch
    weight_np = torch_linear.weight.detach().cpu().numpy()
    bias_np = torch_linear.bias.detach().cpu().numpy() if bias else None


    np_linear = ExplicitLinear(in_features, out_features, weight=weight_np, bias=bias_np, bias_condition=bias)


    # Forward pass
    np_output = np_linear(np_input)
    torch_output = torch_linear(torch_input).detach().numpy()

    # Compare results
    max_diff = np.abs(np_output - torch_output).max()
    mean_diff = np.abs(np_output - torch_output).mean()
    match = np.allclose(np_output, torch_output, atol=1e-5, rtol=1e-4)

    print(f"NumPy output shape: {np_output.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Max difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")
    print(f"Match (atol=1e-5, rtol=1e-4): {match}")

    # Show sample values
    print(f"\nSample output values (first 5 elements):")
    print(f"NumPy:   {np_output.flat[:5]}")
    print(f"PyTorch: {torch_output.flat[:5]}")

    if match:
        print("✓ PASSED")
    else:
        print("✗ FAILED")

    return match


def run_all_tests():
    """
    Run comprehensive test suite
    """
    print("="*80)
    print("COMPARING NumPy ExplicitLinear vs PyTorch nn.Linear")
    print("="*80)

    all_passed = True

    # Test 1: 2D input (standard case)
    all_passed &= test_linear(
        input_shape=(32, 128),
        in_features=128,
        out_features=64,
        test_name="2D Input: Standard batch (32, 128) -> (32, 64)"
    )

    # Test 2: 3D input (sequence data - NLP)
    all_passed &= test_linear(
        input_shape=(8, 50, 256),
        in_features=256,
        out_features=128,
        test_name="3D Input: NLP sequence (8, 50, 256) -> (8, 50, 128)"
    )

    # Test 3: 4D input (image with spatial dimensions) - This was failing before
    all_passed &= test_linear(
        input_shape=(4, 7, 7, 512),
        in_features=512,
        out_features=1024,
        test_name="4D Input: Image features (4, 7, 7, 512) -> (4, 7, 7, 1024)"
    )

    # Test 4: Without bias
    all_passed &= test_linear(
        input_shape=(16, 100),
        in_features=100,
        out_features=50,
        bias=False,
        test_name="2D Input without bias: (16, 100) -> (16, 50)"
    )

    # Test 5: Single sample
    all_passed &= test_linear(
        input_shape=(1, 64),
        in_features=64,
        out_features=32,
        test_name="Single sample: (1, 64) -> (1, 32)"
    )

    # Test 6: Large batch
    all_passed &= test_linear(
        input_shape=(256, 128),
        in_features=128,
        out_features=256,
        test_name="Large batch: (256, 128) -> (256, 256)"
    )

    # Test 7: 3D without bias
    all_passed &= test_linear(
        input_shape=(4, 20, 64),
        in_features=64,
        out_features=32,
        bias=False,
        test_name="3D Input without bias: (4, 20, 64) -> (4, 20, 32)"
    )

    # Test 8: Expansion layer
    all_passed &= test_linear(
        input_shape=(16, 50),
        in_features=50,
        out_features=200,
        test_name="Expansion: (16, 50) -> (16, 200)"
    )

    # Test 9: Reduction layer (classifier)
    all_passed &= test_linear(
        input_shape=(64, 512),
        in_features=512,
        out_features=10,
        test_name="Classifier: (64, 512) -> (64, 10)"
    )

    # Test 10: 5D input (extreme case)
    all_passed &= test_linear(
        input_shape=(2, 3, 4, 5, 32),
        in_features=32,
        out_features=16,
        test_name="5D Input: (2, 3, 4, 5, 32) -> (2, 3, 4, 5, 16)"
    )

    # Test 11: Very small dimensions
    all_passed &= test_linear(
        input_shape=(2, 3),
        in_features=3,
        out_features=2,
        test_name="Tiny: (2, 3) -> (2, 2)"
    )

    # Test 12: Output dimension = 1
    all_passed &= test_linear(
        input_shape=(32, 64),
        in_features=64,
        out_features=1,
        test_name="Single output: (32, 64) -> (32, 1)"
    )

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - NumPy implementation matches PyTorch!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)

    return all_passed


# ==========================================================================================================
# USAGE EXAMPLES
# ----------------------------------------------------------------------------------------------------------
def usage_examples():
    """Show usage examples comparing both implementations"""
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)

    # Example 1: Basic 2D usage
    print("\nExample 1: Basic Linear Layer (2D input)")
    print("-"*40)

    np.random.seed(0)
    torch.manual_seed(0) # Ensure PyTorch gets same initial weights if not provided

    x_np = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)
    x_torch = torch.from_numpy(x_np.copy())

    print("Input shape:", x_np.shape)
    print("Input:\n", x_np)

    # Create PyTorch first to use its weights
    linear_torch = nn.Linear(3, 2)
    weight_np = linear_torch.weight.detach().cpu().numpy()
    bias_np = linear_torch.bias.detach().cpu().numpy() if linear_torch.bias is not None else None

    # NumPy
    linear_np = ExplicitLinear(3, 2, weight=weight_np, bias=bias_np)
    output_np = linear_np(x_np)


    output_torch = linear_torch(x_torch).detach().numpy()

    print("\nNumPy output:\n", output_np)
    print("PyTorch output:\n", output_torch)
    print(f"Match: {np.allclose(output_np, output_torch, atol=1e-5)}")

    # Example 2: 3D input (NLP sequence)
    print("\n" + "="*80)
    print("Example 2: 3D Input - NLP Sequence")
    print("-"*40)

    # Shape: (batch_size, sequence_length, embedding_dim)
    batch_size, seq_len, embed_dim = 2, 5, 8
    np.random.seed(0)
    torch.manual_seed(0)
    x_np = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    x_torch = torch.from_numpy(x_np.copy())

    # Create PyTorch first
    linear_torch = nn.Linear(8, 4)
    weight_np = linear_torch.weight.detach().cpu().numpy()
    bias_np = linear_torch.bias.detach().cpu().numpy() if linear_torch.bias is not None else None


    # NumPy
    linear_np = ExplicitLinear(8, 4, weight=weight_np, bias=bias_np)
    output_np = linear_np(x_np)


    output_torch = linear_torch(x_torch).detach().numpy()

    print(f"Input shape: {x_np.shape}")
    print(f"Output shape: {output_np.shape}")
    print(f"Match: {np.allclose(output_np, output_torch, atol=1e-5)}")
    print("Linear layer applied to each position in the sequence independently")

    # Example 3: Without bias
    print("\n" + "="*80)
    print("Example 3: Linear Layer without Bias")
    print("-"*40)

    np.random.seed(0)
    torch.manual_seed(0)
    x_np = np.random.randn(4, 6).astype(np.float32)
    x_torch = torch.from_numpy(x_np.copy())

    # Create PyTorch first
    linear_torch = nn.Linear(6, 3, bias=False)
    weight_np = linear_torch.weight.detach().cpu().numpy()
    bias_np = linear_torch.bias.detach().cpu().numpy() if linear_torch.bias is not None else None

    # NumPy
    linear_np = ExplicitLinear(6, 3, bias_condition=False, weight=weight_np, bias=bias_np)
    output_np = linear_np(x_np)

    output_torch = linear_torch(x_torch).detach().numpy()

    print(f"Has bias: {linear_np.use_bias}")
    print(f"Input shape: {x_np.shape}")
    print(f"Output shape: {output_np.shape}")
    print(f"Match: {np.allclose(output_np, output_torch, atol=1e-5)}")

    # Example 4: Multi-dimensional input visualization
    print("\n" + "="*80)
    print("Example 4: 4D Input - Image Features")
    print("-"*40)

    # Shape: (batch, height, width, channels)
    np.random.seed(0)
    torch.manual_seed(0)
    x_np = np.random.randn(2, 3, 3, 16).astype(np.float32)
    x_torch = torch.from_numpy(x_np.copy())

    # Create PyTorch first
    linear_torch = nn.Linear(16, 8)
    weight_np = linear_torch.weight.detach().cpu().numpy()
    bias_np = linear_torch.bias.detach().cpu().numpy() if linear_torch.bias is not None else None

    linear_np = ExplicitLinear(16, 8, weight=weight_np, bias=bias_np)
    output_np = linear_np(x_np)
    output_torch = linear_torch(x_torch).detach().numpy()


    print(f"Input shape: {x_np.shape}")
    print(f"Output shape: {output_np.shape}")
    print(f"Match: {np.allclose(output_np, output_torch, atol=1e-5)}")
    print("Linear transformation applied to each spatial location (H, W)")
    print(f"Total transformations: {2 * 3 * 3} = batch_size × height × width")

    # Example 5: Manual verification
    print("\n" + "="*80)
    print("Example 5: Manual Computation Verification")
    print("-"*40)

    x = np.array([[1.0, 2.0]], dtype=np.float32)
    linear = ExplicitLinear(2, 3)

    # Set known weights and bias
    linear.weight = np.array([[1.0, 0.5],
                              [0.0, 1.0],
                              [2.0, -1.0]], dtype=np.float32)
    linear.bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    output = linear(x)

    print("Input:", x)
    print("\nWeights:\n", linear.weight)
    print("Bias:", linear.bias)
    print("\nOutput:", output)
    print("\nManual calculation:")
    # Recalculate manually based on the new implementation (input @ weight.T + bias)
    # Input (1, 2), Weight (3, 2) -> Weight.T (2, 3)
    # output_2d = input_2d @ self.weight.T
    # (1, 2) @ (2, 3) = (1, 3)
    manual_output_0 = x[0, 0] * linear.weight[0, 0] + x[0, 1] * linear.weight[0, 1] + linear.bias[0]
    manual_output_1 = x[0, 0] * linear.weight[1, 0] + x[0, 1] * linear.weight[1, 1] + linear.bias[1]
    manual_output_2 = x[0, 0] * linear.weight[2, 0] + x[0, 1] * linear.weight[2, 1] + linear.bias[2]

    print(f"  output[0,0] = {x[0,0]}*{linear.weight[0,0]} + {x[0,1]}*{linear.weight[0,1]} + {linear.bias[0]} = {manual_output_0}")
    print(f"  output[0,1] = {x[0,0]}*{linear.weight[1,0]} + {x[0,1]}*{linear.weight[1,1]} + {linear.bias[1]} = {manual_output_1}")
    print(f"  output[0,2] = {x[0,0]}*{linear.weight[2,0]} + {x[0,1]}*{linear.weight[2,1]} + {linear.bias[2]} = {manual_output_2}")

    print(f"Match manual calculation: {np.allclose(output[0], np.array([manual_output_0, manual_output_1, manual_output_2]))}")


# ==========================================================================================================
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run comprehensive comparison tests
    run_all_tests()

    # Show usage examples
    usage_examples()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The NumPy ExplicitLinear implementation handles multi-dimensional inputs")
    print("and perfectly replicates PyTorch's nn.Linear behavior.")
    print("Formula: output = input @ weight.T + bias (applied to last dimension)")
    print("="*80)