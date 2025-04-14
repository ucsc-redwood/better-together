import numpy as np
from scipy.sparse import csr_matrix
import os
import shutil


def save_csr_files(dense_file, output_prefix, threshold=0.0):
    """
    Convert dense weight file to CSR format and save the components
    """
    # Read dense weights
    weights = np.loadtxt(dense_file)

    # For convolutional layers, reshape the weights if needed
    if len(weights.shape) == 1:
        if "features_0" in dense_file:  # First conv layer
            weights = weights.reshape(64, 3 * 3 * 3)
        elif "features_3" in dense_file:  # Second conv layer
            weights = weights.reshape(192, 64 * 3 * 3)
        elif "features_6" in dense_file:  # Third conv layer
            weights = weights.reshape(384, 192 * 3 * 3)
        elif "features_8" in dense_file:  # Fourth conv layer
            weights = weights.reshape(256, 384 * 3 * 3)
        elif "features_10" in dense_file:  # Fifth conv layer
            weights = weights.reshape(256, 256 * 3 * 3)
        elif "classifier" in dense_file:  # Linear layer
            weights = weights.reshape(10, -1)

    # Create sparse matrix (CSR format)
    sparse_weights = csr_matrix(weights, dtype=np.float32)

    # Save the CSR components
    np.savetxt(f"{output_prefix}_values.txt", sparse_weights.data, fmt="%.6f")
    np.savetxt(f"{output_prefix}_row_ptr.txt", sparse_weights.indptr, fmt="%d")
    np.savetxt(f"{output_prefix}_col_idx.txt", sparse_weights.indices, fmt="%d")

    # Print sparsity information
    total_elements = weights.size
    nonzero_elements = len(sparse_weights.data)
    sparsity = (total_elements - nonzero_elements) / total_elements * 100
    print(f"File: {dense_file}")
    print(f"Shape: {weights.shape}")
    print(f"Total elements: {total_elements}")
    print(f"Non-zero elements: {nonzero_elements}")
    print(f"Sparsity: {sparsity:.2f}%")
    print("-" * 50)


def copy_and_rename_bias(src_file, dst_file):
    """
    Copy and rename bias files
    """
    shutil.copy2(src_file, dst_file)


def main():
    # Dictionary mapping source files to desired output paths
    weight_mapping = {
        "data/sparse/features_0_weight.txt": "data/sparse/conv1",
        "data/sparse/features_3_weight.txt": "data/sparse/conv2",
        "data/sparse/features_6_weight.txt": "data/sparse/conv3",
        "data/sparse/features_8_weight.txt": "data/sparse/conv4",
        "data/sparse/features_10_weight.txt": "data/sparse/conv5",
        "data/sparse/classifier_weight.txt": "data/sparse/linear",
    }

    bias_mapping = {
        "data/sparse/features_0_bias.txt": "data/sparse/conv1_bias.txt",
        "data/sparse/features_3_bias.txt": "data/sparse/conv2_bias.txt",
        "data/sparse/features_6_bias.txt": "data/sparse/conv3_bias.txt",
        "data/sparse/features_8_bias.txt": "data/sparse/conv4_bias.txt",
        "data/sparse/features_10_bias.txt": "data/sparse/conv5_bias.txt",
        "data/sparse/classifier_bias.txt": "data/sparse/linear_bias.txt",
    }

    # Convert weight files to CSR format
    for input_file, output_prefix in weight_mapping.items():
        if os.path.exists(input_file):
            save_csr_files(input_file, output_prefix)
        else:
            print(f"Warning: Input file not found: {input_file}")

    # Copy and rename bias files
    for src, dst in bias_mapping.items():
        if os.path.exists(src):
            copy_and_rename_bias(src, dst)
            print(f"Copied bias file: {src} -> {dst}")
        else:
            print(f"Warning: Bias file not found: {src}")


if __name__ == "__main__":
    main()
