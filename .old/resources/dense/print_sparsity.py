import numpy as np


# Function to calculate sparsity
def calculate_sparsity(filename):
    # Load data from text file
    data = np.loadtxt(filename)

    # Count non-zero elements
    non_zero_count = np.count_nonzero(data)

    # Total number of elements
    total_count = data.size

    # Calculate sparsity
    sparsity = 1.0 - (non_zero_count / total_count)
    return sparsity


# List of file names
file_names = [
    "features_0_weight.txt",
    "features_3_weight.txt",
    "features_6_weight.txt",
    "features_8_weight.txt",
    "features_10_weight.txt",
    "features_0_bias.txt",
    "features_3_bias.txt",
    "features_6_bias.txt",
    "features_8_bias.txt",
    "features_10_bias.txt",
    "classifier_bias.txt",
    "classifier_weight.txt",
]

# Calculate and print sparsity for each file
for filename in file_names:
    sparsity = calculate_sparsity(filename)
    print(f"{filename}: Sparsity = {sparsity:.2f}")
