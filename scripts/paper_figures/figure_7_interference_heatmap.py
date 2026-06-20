"""
Output paths:
- SVG: scripts/paper_figures/svg/figure_7_interference_heatmap.svg
- PNG: scripts/paper_figures/png/figure_7_interference_heatmap.png
"""

#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Data definition
data = {
    "Google Pixel": {"X1": 1.386754, "A78": 1.202373, "A55": 1.397320, "GPU": 0.862662},
    "OnePlus": {"X3": 1.384255, "A715": 1.005636, "A510": 0.582311, "GPU": 0.639161},
    "Jetson": {"A78AE": 1.428366, "GPU": 1.118529},
    "Jetson (low-power)": {"A78AE": 1.298886, "GPU": 1.174315},
}

# Plot setup
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
sns.set_theme(context="paper", font_scale=2)

# Create colorbar axis
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

# Plot each device
for idx, (device, values) in enumerate(data.items()):
    # Create DataFrame for device
    df = pd.DataFrame(values, index=[0]).T

    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        center=1.0,
        cbar=idx == 0,
        cbar_ax=cbar_ax if idx == 0 else None,
        square=True,
        ax=axes[idx],
    )

    # Customize subplot
    axes[idx].set_title(device, fontsize=20)
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("")
    axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontsize=20)

# Adjust layout
plt.subplots_adjust(right=0.9)

# Save figures
base_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(
    os.path.join(base_dir, "svg", "figure_7_interference_heatmap.svg"),
    bbox_inches="tight",
    format="svg",
)
plt.savefig(
    os.path.join(base_dir, "png", "figure_7_interference_heatmap.png"),
    bbox_inches="tight",
    format="png",
    dpi=300,
)
plt.close()
