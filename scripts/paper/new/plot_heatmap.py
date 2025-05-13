#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Data from the correlation analysis
data = {
    'Google Pixel': {
        'X1': 1.386754,
        'A78': 1.202373, 
        'A55': 1.397320,
        'GPU': 0.862662
    },
    'OnePlus': {
        'X3': 1.384255,
        'A715': 1.005636,
        'A510': 0.582311,
        'GPU': 0.639161
    },
    'Jetson': {
        'A78AE': 1.428366,
        'GPU': 1.118529
    },
    'Jetson (low-power)': {
        'A78AE': 1.298886,
        'GPU': 1.174315
    }
}

# Create figure with 4 subplots in a row
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
sns.set_theme(context='paper',font_scale=2)

# Create a colorbar axis
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

# Plot each device separately
for idx, (device, values) in enumerate(data.items()):
    # Create DataFrame for this device
    df = pd.DataFrame(values, index=[0]).T
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',
                center=1.0,
                cbar=idx==0,  # Only show colorbar for first plot
                cbar_ax=cbar_ax if idx==0 else None,  # Use shared colorbar axis
                square=True,
                ax=axes[idx])
    
    # Customize subplot
    axes[idx].set_title(device, fontsize=20)
    axes[idx].set_xlabel('')
    if idx == 0:  # Only show y-label for first plot
        axes[idx].set_ylabel('')
    else:
        axes[idx].set_ylabel('')
    
    # Set font size for y-axis labels (processor names)
    axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontsize=20)

# Add overall title
# plt.suptitle('Average Performance Ratio Across All Stages', y=1.1, fontsize=14)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.9)  # Make room for colorbar

# Save the plot
plt.savefig('processor_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
