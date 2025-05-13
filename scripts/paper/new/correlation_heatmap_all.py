import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data for the correlation heatmaps
devices = ["OnePlus", "Google", "Jetson", "Jetson (LP)"]
apps = ["CIFAR-D", "CIFAR-S", "Tree"]

# Pearson correlation coefficients
pearson_data = np.array(
    [
        [0.9968, 0.9684, 0.9418],  # OnePlus
        [0.9990, 0.9441, 0.8450],  # Google
        [0.9491, 0.8668, 0.8283],  # Jetson
        [0.9548, 0.8926, 0.8886],  # Jetson (lowpower)
    ]
)

# Calculate row and column averages
row_averages = np.mean(pearson_data, axis=1, keepdims=True)
column_averages = np.mean(pearson_data, axis=0, keepdims=True)
overall_average = np.mean(pearson_data)

# Append averages to the data matrix
pearson_data = np.hstack((pearson_data, row_averages))
pearson_data = np.vstack((pearson_data, np.append(column_averages, overall_average)))

# Update labels to include averages
devices.append("Avg.")
apps.append("Avg.")

# Transpose the data to have devices on x-axis and applications on y-axis
pearson_data_transposed = pearson_data.T

# Set up the figure - taller in height
plt.figure(figsize=(12, 3))

# Set font sizes
plt.rcParams.update({"font.size": 24})

# Create heatmap with transposed data
ax = sns.heatmap(
    pearson_data_transposed,
    annot=True,
    cmap="YlGnBu",
    vmin=0.8,
    vmax=1.0,
    xticklabels=devices,
    yticklabels=apps,
    fmt=".4f",
    annot_kws={"size": 20},
)

# Add titles and labels
ax.set_ylabel("", fontsize=24, labelpad=5)
ax.set_xlabel("", fontsize=24, labelpad=5)

# Set tick label sizes
ax.tick_params(axis='both', which='major', labelsize=20)

# Adjust layout for compact height
plt.tight_layout()

# Save the updated figure
# output_path = "/mnt/data/correlation_heatmap_updated.png"
output_path = "./correlation_heatmap_updated.png"
plt.savefig(output_path, bbox_inches="tight", dpi=300)

# Display the saved file path
output_path
