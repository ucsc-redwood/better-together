import numpy as np
import matplotlib.pyplot as plt

# Baselines (Google Pixel)

dense_cpu_baseline = 734
dense_gpu_baseline = 6.76

sparse_cpu_baseline = 29.0
sparse_gpu_baseline = 13.2

tree_cpu_baseline = 15.9
tree_gpu_baseline = 46.3

best_measured_dense = 6.33
best_measured_sparse = 4.05
best_measured_tree = 2.45

# best_measured_dense = 6.72
# best_measured_sparse = 7.04
# best_measured_tree = 2.48

# Google Pixel Dense

# Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
# --------------------------------------------------------------------------------
# SCH-G7L2-G260-6259             :         6.33            6.72           -5.76%
# SCH-G7B2-G519-9320             :         6.24            6.72           -7.13%
# SCH-G7M2-G527-f560             :         6.37            6.72           -5.18%
# SCH-G7B1L1-G578-28c7           :         5.88            6.72          -12.54%
# SCH-G7M1L1-G586-2c3d           :         5.33            6.72          -20.73%
# SCH-G7B1M1-G613-75f7           :         5.72            6.72          -14.89%
# SCH-G7L1M1-G613-5fbd           :         5.02            6.72          -25.36%
# SCH-G7L1B1-G613-5736           :         6.40            6.72           -4.78%
# SCH-G7M1B1-G613-f5ce           :         5.77            6.72          -14.09%
# SCH-G8L1-G574-9715             :         6.75            6.95           -2.85%
# SCH-G8M1-G636-0069             :         6.68            6.95           -3.97%
# SCH-G8B1-G636-b1c2             :         6.67            6.95           -4.00%
# SCH-G9-G000-ffbd               :         9.54            7.68          +24.16%
# SCH-B1G8-G4972-78b4            :         7.49           56.65          -86.79%
# SCH-B1L1G7-G5007-f6ac          :         6.30           56.65          -88.88%
# SCH-M6L3-G498-7a3d             :       506.41          718.95          -29.56%
# SCH-B6L3-G4205-301b            :       530.06          761.00          -30.35%
# SCH-M9-G000-c124               :       769.45         1072.83          -28.28%
# SCH-B9-G000-1612               :       779.17         1119.94          -30.43%
# SCH-L9-G000-3096               :      1863.07         2460.65          -24.29%

# Best-case measured time for dense (ms)


# Compute dense speedups relative to slowest processor
dense_slowest = max(dense_cpu_baseline, dense_gpu_baseline)
dense_speedups = [
    dense_slowest / dense_cpu_baseline,  # CPU speedup over slowest
    dense_slowest / dense_gpu_baseline,  # GPU speedup over slowest
    dense_slowest / best_measured_dense,  # BT speedup over slowest
]

sparse_slowest = max(sparse_cpu_baseline, sparse_gpu_baseline)
sparse_speedups = [
    sparse_slowest / sparse_cpu_baseline,  # CPU speedup over slowest
    sparse_slowest / sparse_gpu_baseline,  # GPU speedup over slowest
    sparse_slowest / best_measured_sparse,  # BT speedup over slowest
]

tree_slowest = max(tree_cpu_baseline, tree_gpu_baseline)
tree_speedups = [
    tree_slowest / tree_cpu_baseline,  # CPU speedup over slowest
    tree_slowest / tree_gpu_baseline,  # GPU speedup over slowest
    tree_slowest / best_measured_tree,  # BT speedup over slowest
]

# Combine into a data matrix: rows = workloads, cols = methods
speedup_data = np.array([dense_speedups, sparse_speedups, tree_speedups])

workloads = ["CIFAR-Dense", "CIFAR-Sparse", "Octree"]
methods = ["CPU", "GPU", "This work"]

# Plot
x = np.arange(len(workloads))
width = 0.25

# Academic color scheme
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
]  # Blue for CPU, Orange for GPU, Green for our work

fig, ax = plt.subplots(figsize=(8, 5))

for i, method in enumerate(methods):
    ax.bar(x + i * width, speedup_data[:, i], width, label=method, color=colors[i])

ax.set_xticks(x + width)
ax.set_xticklabels(workloads)
ax.set_ylabel("Speedup over Slowest Processor")
ax.set_yscale("log")  # Set y-axis to logarithmic scale
ax.set_title("Speedup Comparison Across Workloads", pad=20)
ax.legend(framealpha=0.9)

# Annotate each bar with its value
for i in range(len(workloads)):
    for j in range(len(methods)):
        height = speedup_data[i, j]
        ax.text(x[i] + j * width, height, f"{height:.2f}", ha="center", va="bottom")

ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig("headline_result.png", dpi=300, bbox_inches="tight")
