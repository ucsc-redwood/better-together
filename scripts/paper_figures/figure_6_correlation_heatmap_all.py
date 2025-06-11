"""
Output paths:
- SVG: scripts/paper_figures/svg/figure_6_correlation_heatmap_all.svg
- PNG: scripts/paper_figures/png/figure_6_correlation_heatmap_all.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data definition
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

# Calculate averages
row_averages = np.mean(pearson_data, axis=1, keepdims=True)
column_averages = np.mean(pearson_data, axis=0, keepdims=True)
overall_average = np.mean(pearson_data)

# Append averages to data
pearson_data = np.hstack((pearson_data, row_averages))
pearson_data = np.vstack((pearson_data, np.append(column_averages, overall_average)))

# Update labels
devices.append("Avg.")
apps.append("Avg.")

# Transpose data for visualization
pearson_data_transposed = pearson_data.T

# Plot setup
plt.figure(figsize=(12, 3))
plt.rcParams.update({"font.size": 24})

# Create heatmap
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

# Customize plot
ax.set_ylabel("", fontsize=24, labelpad=5)
ax.set_xlabel("", fontsize=24, labelpad=5)
ax.tick_params(axis="both", which="major", labelsize=20)

# Adjust layout
plt.tight_layout()

# Save figures
base_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(
    os.path.join(base_dir, "svg", "figure_6_correlation_heatmap_all.svg"),
    bbox_inches="tight",
    format="svg",
)
plt.savefig(
    os.path.join(base_dir, "png", "figure_6_correlation_heatmap_all.png"),
    bbox_inches="tight",
    format="png",
    dpi=300,
)
plt.close()
