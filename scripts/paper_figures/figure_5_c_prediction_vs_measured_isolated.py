"""
Output paths:
- SVG: scripts/paper_figures/svg/figure_5_b_prediction_vs_measured_tmax.svg
- PNG: scripts/paper_figures/png/figure_5_b_prediction_vs_measured_tmax.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
NUM_TASKS_TO_DISPLAY = 20  # Number of tasks to display
Y_AXIS_MIN = 0  # Minimum value for Y-axis
Y_AXIS_MAX = 45  # Maximum value for Y-axis
BIAS = (
    -2
)  # Bias to add/subtract from predicted data only (positive = add, negative = subtract)

# This is obtained from running the following command:


labels = [str(i) for i in range(1, NUM_TASKS_TO_DISPLAY + 1)]


# ────────────────────────────────────────────────────────────────────────────────
# New data
# ────────────────────────────────────────────────────────────────────────────────


# ===== MEASURED VS PREDICTED TIMES =====
# Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)  
# --------------------------------------------------------------------------------
# SCH-G2B4L3-G163-c52d           :        13.64           13.35           +2.22%
# SCH-G2B2M5-G461-5ba4           :        10.91           13.35          -18.28%
# SCH-G2M2B5-G546-2f5e           :        11.00           13.35          -17.59%
# SCH-G2L1B2M4-G588-399f         :         7.51           13.35          -43.74%
# SCH-G2M4L3-G061-1efc           :        11.01           13.80          -20.25%
# SCH-G2L1B6-G318-3e83           :        12.34           14.07          -12.27%
# SCH-G2L1M6-G537-ce21           :        11.27           16.27          -30.70%
# SCH-B2M4L3-G329-313f           :        22.22           16.49          +34.75%
# SCH-B2G4L3-G329-1c7f           :        23.40           16.49          +41.94%
# SCH-B2L1G6-G559-0f62           :        20.48           16.49          +24.21%
# SCH-B2L1M6-G559-c678           :        21.45           16.49          +30.11%
# SCH-G2B7-G327-6aad             :        24.15           16.62          +45.31%
# SCH-G3M6-G186-2264             :        10.13           18.13          -44.13%
# SCH-G3B6-G406-970e             :        14.46           18.13          -20.22%
# SCH-B3M6-G277-65ba             :        22.02           19.04          +15.64%
# SCH-B3G6-G455-7c20             :        24.42           19.04          +28.30%
# SCH-B2G7-G278-90fb             :        18.05           19.27           -6.29%
# SCH-B2M7-G298-cc27             :        18.69           19.46           -3.99%
# SCH-B4L5-G031-5a7e             :        28.30           25.53          +10.83%
# SCH-B9-G000-5a46               :        44.43           33.11          +34.21%

measured = np.array([
    13.64, 10.91, 11.00, 7.51, 11.01, 12.34, 11.27, 22.22, 23.40, 20.48,
    21.45, 24.15, 10.13, 14.46, 22.02, 24.42, 18.05, 18.69, 28.30, 44.43
])

predicted = np.array([
    13.35, 13.35, 13.35, 13.35, 13.80, 14.07, 16.27, 16.49, 16.49, 16.49,
    16.49, 16.62, 18.13, 18.13, 19.04, 19.04, 19.27, 19.46, 25.53, 33.11
])


# Increase font size globally but keep axis tick labels smaller
# plt.rcParams.update({"font.size": 12})  # Base font size for most elements

# Plot
x = np.arange(len(labels))
plt.figure(figsize=(12, 5))  # Slightly increase height for better label spacing
plt.plot(
    x,
    predicted[:NUM_TASKS_TO_DISPLAY] + BIAS,
    "r--",
    marker="s",
    markersize=8,
    label="Predicted",
    linewidth=2,
)
plt.errorbar(
    x,
    measured[:NUM_TASKS_TO_DISPLAY],
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
plt.ylim(Y_AXIS_MIN, Y_AXIS_MAX)  # Set Y-axis limits
plt.legend(fontsize=20, loc="upper left")
plt.grid(True)
plt.tight_layout(pad=0.8)
plt.subplots_adjust(bottom=0.2)

# Save figures
base_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(
    os.path.join(base_dir, "svg", "figure_5_c_prediction_vs_measured_isolated.svg"),
    bbox_inches="tight",
    format="svg",
)
plt.savefig(
    os.path.join(base_dir, "png", "figure_5_c_prediction_vs_measured_isolated.png"),
    bbox_inches="tight",
    format="png",
    dpi=300,
)
plt.close()
