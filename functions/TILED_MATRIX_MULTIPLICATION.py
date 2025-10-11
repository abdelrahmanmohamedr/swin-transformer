############################################################################################################
# LINEAR LAYER WITH TILED MATRIX MULTIPLICATION (Hardware-Optimized)
############################################################################################################

import numpy as np
import torch
import torch.nn as nn
import math

# ==========================================================================================================
# NUMPY IMPLEMENTATION WITH TILED MATRIX MULTIPLICATION
# ----------------------------------------------------------------------------------------------------------

class ExplicitLinear:
    def __init__(self, in_features, out_features, weight=None, bias=None, bias_condition=True,
                 ci=32, co=32, m=49):
        """
        NumPy-compatible Linear layer with tiled matrix multiplication.
        
        Tiling parameters:
        - ci: Column tile size for input A (default 32)
        - co: Column tile size for weight B (default 32) 
        - m: Row tile size for input A, equals number of parallel multipliers per PE (default 49)
        """
        self.in_features = in_features
        self.out_features = out_features
        
        # Tiling configuration
        self.ci = ci  # Input column tile size (depth)
        self.co = co  # Output column tile size (PE count)
        self.m = m    # Row tile size (parallel multipliers per PE)

        # Store as numpy for compatibility
        if weight is not None:
            assert weight.shape == (out_features, in_features)
            self.weight = weight.astype(np.float32)
        else:
            weight_tensor = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(weight_tensor, a=math.sqrt(5))
            self.weight = weight_tensor.numpy().astype(np.float32)

        if bias is None and bias_condition is True:
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

    def tiled_matmul(self, A, B):
        """
        Tiled matrix multiplication: C = A @ B^T
        
        A shape: (M, K) - Input data (M samples, K features)
        B shape: (N, K) - Weight matrix (N outputs, K features)
        C shape: (M, N) - Output
        
        Tiling:
        - A is tiled into (m x ci) tiles
        - B is tiled into (ci x co) tiles
        - 32 PEs work in parallel, each PE has 49 parallel multipliers
        """
        M, K = A.shape
        N, K_B = B.shape
        assert K == K_B, "Matrix dimensions must match"
        
        # Initialize output
        C = np.zeros((M, N), dtype=np.float32)
        
        # Calculate number of tiles
        num_row_tiles = (M + self.m - 1) // self.m      # Tiles along M dimension
        num_col_tiles = (N + self.co - 1) // self.co    # Tiles along N dimension  
        num_depth_tiles = (K + self.ci - 1) // self.ci  # Tiles along K dimension
        
        print(f"\n=== Tiled Matrix Multiplication ===")
        print(f"Input A: {M} x {K}")
        print(f"Weight B: {N} x {K}")
        print(f"Output C: {M} x {N}")
        print(f"Tile configuration: m={self.m}, ci={self.ci}, co={self.co}")
        print(f"Number of tiles: {num_row_tiles} row tiles × {num_col_tiles} col tiles × {num_depth_tiles} depth tiles")
        
        # Iterate over output tiles (M x N)
        for row_tile_idx in range(num_row_tiles):
            # Calculate row range for this tile
            row_start = row_tile_idx * self.m
            row_end = min(row_start + self.m, M)
            actual_m = row_end - row_start
            
            for col_tile_idx in range(num_col_tiles):
                # Calculate column range for this tile
                col_start = col_tile_idx * self.co
                col_end = min(col_start + self.co, N)
                actual_co = col_end - col_start
                
                # Accumulator for this output tile
                tile_result = np.zeros((actual_m, actual_co), dtype=np.float32)
                
                # Iterate over depth tiles (K dimension)
                for depth_tile_idx in range(num_depth_tiles):
                    depth_start = depth_tile_idx * self.ci
                    depth_end = min(depth_start + self.ci, K)
                    actual_ci = depth_end - depth_start
                    
                    # Extract tiles
                    A_tile = A[row_start:row_end, depth_start:depth_end]  # Shape: (actual_m, actual_ci)
                    B_tile = B[col_start:col_end, depth_start:depth_end]  # Shape: (actual_co, actual_ci)
                    
                    # Simulate hardware computation with 32 PEs
                    # Each PE processes one output column
                    # Each PE has 49 parallel multipliers
                    
                    # B_tile columns are distributed across PEs
                    # PE i processes column i of output (row i of B_tile)
                    for pe_idx in range(actual_co):  # PE index (up to 32 PEs)
                        # Get the weight column for this PE (row of B_tile)
                        weight_col = B_tile[pe_idx, :]  # Shape: (actual_ci,)
                        
                        # Each PE has 49 parallel multipliers
                        # Process all rows of A_tile in parallel with this weight column
                        
                        # Simulate cycle-by-cycle computation
                        pe_result = np.zeros(actual_m, dtype=np.float32)
                        
                        for cycle in range(actual_ci):  # 32 cycles for ci=32
                            # In each cycle, multiply all rows by one weight element
                            # All 49 multipliers work in parallel
                            weight_element = weight_col[cycle]
                            
                            for row_idx in range(actual_m):  # Up to 49 rows processed in parallel
                                # Multiply and accumulate
                                pe_result[row_idx] += A_tile[row_idx, cycle] * weight_element
                        
                        # Store result from this PE (one output column)
                        tile_result[:, pe_idx] += pe_result
                
                # Write tile result to output
                C[row_start:row_end, col_start:col_end] = tile_result
                
        print(f"Tiled computation complete!\n")
        return C

    def forward(self, input_data):
        """
        Forward pass using tiled matrix multiplication.
        """
        # Ensure input is float32 numpy array
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()

        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # Handle multi-dimensional inputs by reshaping to (N, in_features)
        original_shape = input_data.shape
        input_2d = input_data.reshape(-1, self.in_features)
        num_samples = input_2d.shape[0]

        # Perform tiled matrix multiplication: output = input @ weight.T
        # weight shape: (out_features, in_features)
        # We compute: input_2d @ weight.T
        output_2d = self.tiled_matmul(input_2d, self.weight)

        # Add bias if present
        if self.use_bias:
            output_2d += self.bias  # Broadcasting across all samples

        # Reshape back to original shape but with the last dimension replaced by out_features
        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = output_2d.reshape(new_shape)

        return output

    def __call__(self, input_data):
        return self.forward(input_data)


# ==========================================================================================================
# TEST CASES
# ----------------------------------------------------------------------------------------------------------
def test_tiled_linear(input_shape, in_features, out_features, ci=32, co=32, m=49, 
                      bias=True, test_name=""):
    """
    Test tiled implementation against PyTorch
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    print(f"Input shape: {input_shape}")
    print(f"Input features: {in_features}, Output features: {out_features}")
    print(f"Tile config: m={m} (rows), ci={ci} (depth), co={co} (PEs)")
    print(f"Bias: {bias}")
    print(f"{'-'*80}")

    # Create random input
    np.random.seed(42)
    torch.manual_seed(42)

    np_input = np.random.randn(*input_shape).astype(np.float32)
    torch_input = torch.from_numpy(np_input.copy())

    # Create PyTorch reference
    torch_linear = nn.Linear(in_features, out_features, bias=bias)
    weight_np = torch_linear.weight.detach().cpu().numpy()
    bias_np = torch_linear.bias.detach().cpu().numpy() if bias else None

    # Create tiled implementation
    np_linear = ExplicitLinear(in_features, out_features, 
                               weight=weight_np, bias=bias_np, bias_condition=bias,
                               ci=ci, co=co, m=m)

    # Forward pass
    np_output = np_linear(np_input)
    torch_output = torch_linear(torch_input).detach().numpy()

    # Compare results
    max_diff = np.abs(np_output - torch_output).max()
    mean_diff = np.abs(np_output - torch_output).mean()
    match = np.allclose(np_output, torch_output, atol=1e-5, rtol=1e-4)

    print(f"\nResults:")
    print(f"NumPy output shape: {np_output.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Max difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")
    print(f"Match (atol=1e-5, rtol=1e-4): {match}")

    if match:
        print("✓ PASSED")
    else:
        print("✗ FAILED")

    return match


# ==========================================================================================================
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*80)
    print("TILED MATRIX MULTIPLICATION - LINEAR LAYER")
    print("Hardware Configuration: 32 PEs × 49 Parallel Multipliers")
    print("="*80)

    # Test 1: Standard case matching your example dimensions
    test_tiled_linear(
        input_shape=(2916, 96),
        in_features=96,
        out_features=128,
        ci=32, co=32, m=49,
        test_name="Example case: (2916, 96) -> (2916, 128)"
    )

    # Test 2: Smaller example for verification
    test_tiled_linear(
        input_shape=(98, 64),
        in_features=64,
        out_features=96,
        ci=32, co=32, m=49,
        test_name="Small case: (98, 64) -> (98, 96)"
    )

    # Test 3: 3D input (batch processing)
    test_tiled_linear(
        input_shape=(8, 50, 96),
        in_features=96,
        out_features=128,
        ci=32, co=32, m=49,
        test_name="3D Input: (8, 50, 96) -> (8, 50, 128)"
    )

    # Test 4: Without bias
    test_tiled_linear(
        input_shape=(147, 96),
        in_features=96,
        out_features=128,
        ci=32, co=32, m=49,
        bias=False,
        test_name="No bias: (147, 96) -> (147, 128)"
    )

    print("\n" + "="*80)
    print("TILED COMPUTATION SUMMARY")
    print("="*80)
    print("✓ Input A tiled vertically by ci=32, horizontally by m=49")
    print("✓ Weight B tiled vertically by ci=32, horizontally by co=32")
    print("✓ 32 PEs process in parallel (one PE per output column)")
    print("✓ Each PE has 49 parallel multipliers")
    print("✓ Computation cycles: ci=32 (one cycle per depth element)")
    print("="*80)
