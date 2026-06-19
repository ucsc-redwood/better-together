"""
Output paths:
- SVG: scripts/paper_figures/svg/figure_4_overall_speedup.svg
- PNG: scripts/paper_figures/png/figure_4_overall_speedup.png
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Data updated with new baseline results (omp = CPU, cu/vk = GPU, tree = Octree)
data = {
    "3A021JEHN02756": {
        "CIFAR-dense": {"CPU baseline": 940, "GPU baseline": 11.4},
        "CIFAR-sparse": {"CPU baseline": 45.8, "GPU baseline": 44.9},
        "Octree": {"CPU baseline": 14.2, "GPU baseline": 58.7},
    },
    "9b034f1b": {
        "CIFAR-dense": {"CPU baseline": 730, "GPU baseline": 12.1},
        "CIFAR-sparse": {"CPU baseline": 53.2, "GPU baseline": 27.9},
        "Octree": {"CPU baseline": 12.7, "GPU baseline": 47.2},
    },
    "jetson": {
        "CIFAR-dense": {"CPU baseline": 23.5, "GPU baseline": 5.48},
        "CIFAR-sparse": {"CPU baseline": 486, "GPU baseline": 27.2},
        "Octree": {"CPU baseline": 16.2, "GPU baseline": 5.42},
    },
    "jetsonlowpower": {
        "CIFAR-dense": {"CPU baseline": 58.5, "GPU baseline": 23.6},
        "CIFAR-sparse": {"CPU baseline": 1042, "GPU baseline": 101},
        "Octree": {"CPU baseline": 39.7, "GPU baseline": 7.28},
    },
}

measured_data = {
    "3A021JEHN02756": {
        "CIFAR-dense": 5.89,
        "CIFAR-sparse": 6.21,
        "Octree": 1.87,
    },
    "9b034f1b": {
        "CIFAR-dense": 6.74,
        "CIFAR-sparse": 6.68,
        "Octree": 2.03,
    },
    "jetson": {"CIFAR-dense": 5.27, "CIFAR-sparse": 23.86, "Octree": 5.02},
    "jetsonlowpower": {"CIFAR-dense": 21.85, "CIFAR-sparse": 56.74, "Octree": 9.84},
}

# Process data to calculate actual speedups (baseline_time / optimized_time)
processed_data = {}
baseline_choices = {}
for device, apps in data.items():
    processed_data[device] = {}
    baseline_choices[device] = {}
    for app, baselines in apps.items():
        cpu_val = baselines["CPU baseline"]
        gpu_val = baselines["GPU baseline"]
        optimized_time = measured_data[device][app]

        # Calculate speedups for both baselines
        cpu_speedup = cpu_val / optimized_time
        gpu_speedup = gpu_val / optimized_time

        # Choose the better baseline (the one that gives minimum speedup - strongest baseline)
        if gpu_speedup < cpu_speedup:
            best_speedup = gpu_speedup
            baseline_choices[device][app] = "GPU"
        else:
            best_speedup = cpu_speedup
            baseline_choices[device][app] = "CPU"
        processed_data[device][app] = best_speedup

# Prepare data for plotting
devices = list(processed_data.keys())
apps = ["CIFAR-dense", "CIFAR-sparse", "Octree"]
device_names = {
    "3A021JEHN02756": "Google Pixel",
    "9b034f1b": "OnePlus",
    "jetson": "Jetson",
    "jetsonlowpower": "Jetson (Low Power)",
}

# Define colors for academic papers - colorblind-friendly palette
app_colors = {
    "CIFAR-dense": "#0173B2",  # blue
    "CIFAR-sparse": "#DE8F05",  # orange
    "Octree": "#029E73",  # green
}

# Set up the figure with a clean, academic style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(devices))
width = 0.25
multiplier = 0

# Hatching patterns for CPU vs GPU baselines
hatch_patterns = {"CPU": "///", "GPU": "..."}

# Plot each application's speedup for each device
for app in apps:
    speedups = []
    hatches = []
    for device in devices:
        speedups.append(processed_data[device][app])
        hatches.append(hatch_patterns[baseline_choices[device][app]])

    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        speedups,
        width,
        label=app,
        color=app_colors[app],
        edgecolor="black",
        linewidth=0.5,
    )

    # Apply hatching based on baseline choice
    for i, rect in enumerate(rects):
        rect.set_hatch(hatches[i])

    # Add value labels on top of bars
    for i, rect in enumerate(rects):
        height = rect.get_height()
        label_text = f"{speedups[i]:.2f}"
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + max(speedups) * 0.02,  # Dynamic offset based on max value
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    multiplier += 1

# Add labels and title
# ax.set_title("Speedup vs. Best Baseline by Device and Application")
ax.set_xticks(x + width)
ax.set_xticklabels([device_names.get(d, d) for d in devices])
ax.set_ylabel("Speedup (x)")
ax.set_ylim(0, max([processed_data[device][app] for device in devices for app in apps]) * 1.2)

# Add a grid for better readability
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Create a combined legend showing both application colors and baseline patterns
legend_elements = []
for app in apps:
    # Add entries for both CPU and GPU baselines for each application
    legend_elements.append(
        mpatches.Patch(
            facecolor=app_colors[app],
            edgecolor="black",
            hatch=hatch_patterns["CPU"],
            label=f"{app} (vs. CPU)",
        )
    )
    legend_elements.append(
        mpatches.Patch(
            facecolor=app_colors[app],
            edgecolor="black",
            hatch=hatch_patterns["GPU"],
            label=f"{app} (vs. GPU)",
        )
    )

# Add the combined legend
ax.legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    ncol=1,
    frameon=True,
    fontsize=12,
)

# Create a table showing which baseline was chosen for each device-app combination
table_data = []
for device in devices:
    row = [device_names.get(device, device)]
    for app in apps:
        row.append(baseline_choices[device][app])
    table_data.append(row)

table_columns = ["Device"] + apps
print("\nBaseline chosen for each device-application combination:")
for i, row in enumerate(table_data):
    print(f"{row[0]}: {apps[0]}: {row[1]}, {apps[1]}: {row[2]}, {apps[2]}: {row[3]}")

print("\nDetailed speedup calculations:")
print("-" * 80)
for device in devices:
    print(f"\n{device_names.get(device, device)}:")
    for app in apps:
        cpu_baseline = data[device][app]["CPU baseline"]
        gpu_baseline = data[device][app]["GPU baseline"]
        optimized_time = measured_data[device][app]
        cpu_speedup = cpu_baseline / optimized_time
        gpu_speedup = gpu_baseline / optimized_time
        chosen_baseline = baseline_choices[device][app]
        best_speedup = processed_data[device][app]

        print(f"  {app}:")
        print(
            f"    CPU baseline: {cpu_baseline:.1f}ms, GPU baseline: {gpu_baseline:.1f}ms, Optimized: {optimized_time:.2f}ms"
        )
        print(f"    CPU speedup: {cpu_speedup:.1f}x, GPU speedup: {gpu_speedup:.1f}x")
        print(f"    Best baseline: {chosen_baseline}, Best speedup: {best_speedup:.1f}x")

# Calculate and print geometric means and maximums
print("\nSpeedup Statistics:")
print("-" * 50)

# Calculate for each device
for device in devices:
    device_speedups = [processed_data[device][app] for app in apps]
    geo_mean = np.exp(np.mean(np.log(device_speedups)))
    max_speedup = max(device_speedups)
    print(f"{device_names.get(device, device)}:")
    print(f"  Geometric Mean: {geo_mean:.2f}")
    print(f"  Maximum: {max_speedup:.2f}")

# Calculate overall statistics
all_speedups = [processed_data[device][app] for device in devices for app in apps]
overall_geo_mean = np.exp(np.mean(np.log(all_speedups)))
overall_max = max(all_speedups)
print("\nOverall Statistics:")
print(f"  Geometric Mean: {overall_geo_mean:.2f}")
print(f"  Maximum: {overall_max:.2f}")

plt.tight_layout()
base_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(
    os.path.join(base_dir, "svg", "figure_4_overall_speedup.svg"),
    bbox_inches="tight",
    format="svg",
)
plt.savefig(
    os.path.join(base_dir, "png", "figure_4_overall_speedup.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
