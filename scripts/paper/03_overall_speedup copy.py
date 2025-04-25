import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Data as provided
data = {
    "3A021JEHN02756": {
        "cifar-dense": {"CPU baseline": 155.63, "GPU baseline": 1.89},
        "cifar-sparse": {"CPU baseline": 8.51, "GPU baseline": 8.35},
        "tree": {"CPU baseline": 8.40, "GPU baseline": 34.73},
    },
    "9b034f1b": {
        "cifar-dense": {"CPU baseline": 113.88, "GPU baseline": 1.89},
        "cifar-sparse": {"CPU baseline": 7.52, "GPU baseline": 3.95},
        "tree": {"CPU baseline": 5.99, "GPU baseline": 22.26},
    },
    "jetson": {
        "cifar-dense": {"CPU baseline": 19.90, "GPU baseline": 1.04},
        "cifar-sparse": {"CPU baseline": 4.81, "GPU baseline": 1.14},
        "tree": {"CPU baseline": 3.29, "GPU baseline": 1.08},
    },
    "jetsonlowpower": {
        "cifar-dense": {"CPU baseline": 11.36, "GPU baseline": 1.08},
        "cifar-sparse": {"CPU baseline": 4.58, "GPU baseline": 1.78},
        "tree": {"CPU baseline": 4.26, "GPU baseline": 0.74},
    },
}

# Process data to get best baseline and determine which baseline was chosen
processed_data = {}
baseline_choices = {}
for device, apps in data.items():
    processed_data[device] = {}
    baseline_choices[device] = {}
    for app, baselines in apps.items():
        cpu_val = baselines["CPU baseline"]
        gpu_val = baselines["GPU baseline"]
        # Choose the better baseline (lower speedup means it's a stronger baseline)
        if gpu_val < cpu_val:
            best_baseline = gpu_val
            baseline_choices[device][app] = "GPU"
        else:
            best_baseline = cpu_val
            baseline_choices[device][app] = "CPU"
        processed_data[device][app] = best_baseline

# Prepare data for plotting
devices = list(processed_data.keys())
apps = ["cifar-dense", "cifar-sparse", "tree"]
device_names = {
    "3A021JEHN02756": "Intel",
    "9b034f1b": "AMD",
    "jetson": "Jetson",
    "jetsonlowpower": "Jetson LP",
}

# Define colors for academic papers - colorblind-friendly palette
app_colors = {
    "cifar-dense": "#0173B2",  # blue
    "cifar-sparse": "#DE8F05",  # orange
    "tree": "#029E73",  # green
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

fig, ax = plt.subplots(figsize=(8, 6))
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
        label_text = f"{speedups[i]:.2f}×"
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + 0.1,
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    multiplier += 1

# Add labels and title
ax.set_title("Speedup vs. Best Baseline by Device and Application")
ax.set_xticks(x + width)
ax.set_xticklabels([device_names.get(d, d) for d in devices])
ax.set_ylabel("Speedup (×)")
ax.set_ylim(
    0, max([processed_data[device][app] for device in devices for app in apps]) * 1.2
)

# Add a grid for better readability
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Legend for applications
app_legend = ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True
)

# Create custom legend for hatching patterns
hatch_patches = [
    mpatches.Patch(
        facecolor="lightgray",
        edgecolor="black",
        hatch=hatch_patterns["CPU"],
        label="vs. CPU",
    ),
    mpatches.Patch(
        facecolor="lightgray",
        edgecolor="black",
        hatch=hatch_patterns["GPU"],
        label="vs. GPU",
    ),
]
ax.legend(handles=hatch_patches, loc="upper right", frameon=True)

# Add app legend back after creating the hatch legend
ax.add_artist(app_legend)

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

plt.tight_layout()
plt.savefig("overall_speedup.pdf", format="pdf", bbox_inches="tight")
plt.savefig("overall_speedup.png", dpi=300, bbox_inches="tight")
plt.show()
