import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data for the correlation heatmaps
devices = ["OnePlus", "Google", "Jetson", "Jetson (lowpower)"]
apps = ["CifarDense", "CifarSparse", "Tree"]

# Pearson correlation coefficients
pearson_data = np.array([
    [0.9968, 0.9684, 0.9418],  # 9b034f1b (OnePlus)
    [0.9990, 0.9441, 0.8450],  # 3A021JEHN02756 (Google)
    [0.9491, 0.8668, 0.8283],  # jetson
    [0.9548, 0.8926, 0.8886],  # jetsonlowpower
])

# Transpose the data to have devices on x-axis and applications on y-axis
pearson_data_transposed = pearson_data.T

# Calculate geometric mean
pearson_geomean = np.exp(np.mean(np.log(pearson_data)))

# Set up the figure - wider and shorter
plt.figure(figsize=(12, 4))

# Set font sizes
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'axes.labelsize': 14})

# Create heatmap with transposed data
ax = sns.heatmap(pearson_data_transposed, annot=True, cmap="YlGnBu", vmin=0.8, vmax=1.0, 
            xticklabels=devices, yticklabels=apps, fmt=".4f", annot_kws={"size": 14})
            
# Add titles and labels
ax.set_title('Pearson Correlation Coefficient', fontsize=18, pad=20)
ax.set_xlabel('Devices', fontsize=16, labelpad=10)
ax.set_ylabel('Applications', fontsize=16, labelpad=10)

# Adjust tick labels
plt.xticks(fontsize=14, rotation=0, ha='center')
plt.yticks(fontsize=14, rotation=0)

# # Add geometric mean information
# plt.figtext(0.5, 0.01, f"Geometric Mean: {pearson_geomean:.4f}", 
#             ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300)
print("Updated correlation heatmap saved as 'correlation_heatmap.png'")

# Show the heatmap
plt.show() 

print(pearson_geomean)