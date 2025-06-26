"""
Output paths:
- SVG: scripts/paper_figures/svg/figure_5_a_prediction_vs_measured.svg
- PNG: scripts/paper_figures/png/figure_5_a_prediction_vs_measured.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os


labels = [str(i) for i in range(1, 21)]

measured = np.array(
    [
        5.34,
        5.38,
        4.23,
        3.96,
        7.67,
        5.35,
        6.99,
        5.48,
        5.86,
        7.37,
        8.38,
        15.17,
        33.44,
        15.01,
        14.12,
        21.79,
        22.17,
        26.72,
        30.19,
        68.61,
    ]
)
predicted = np.array(
    [
        7.65,
        7.86,
        7.86,
        7.86,
        9.95,
        9.95,
        9.95,
        9.95,
        9.95,
        9.95,
        11.95,
        15.74,
        19.39,
        19.48,
        20.00,
        30.17,
        30.17,
        38.38,
        38.81,
        108.77,
    ]
)

# Increase font size globally but keep axis tick labels smaller
# plt.rcParams.update({"font.size": 12})  # Base font size for most elements

# Plot
x = np.arange(len(labels))
plt.figure(figsize=(12, 5))  # Slightly increase height for better label spacing
plt.plot(x, predicted, "r--", marker="s", markersize=8, label="Predicted", linewidth=2)
plt.errorbar(
    x,
    measured,
    yerr=0.5,
    fmt="b-",
    marker="^",
    markersize=8,
    label="Measured (mean)",
    linewidth=2,
)

plt.xticks(x, labels, ha="right", fontsize=22)
plt.ylabel("Time (Execution ms)", fontsize=24)
plt.yticks(fontsize=22)  # Set y-axis tick label font size
plt.legend(fontsize=24)
plt.grid(True)
plt.tight_layout(pad=0.8)
plt.subplots_adjust(bottom=0.2)

# Save figures
base_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(
    os.path.join(base_dir, "svg", "figure_5_a_prediction_vs_measured.svg"),
    bbox_inches="tight",
    format="svg",
)
plt.savefig(
    os.path.join(base_dir, "png", "figure_5_a_prediction_vs_measured.png"),
    bbox_inches="tight",
    format="png",
    dpi=300,
)
plt.close()
