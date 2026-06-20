"""
Output paths:
- SVG: scripts/paper_figures/svg/figure_5_a_prediction_vs_measured.svg
- PNG: scripts/paper_figures/png/figure_5_a_prediction_vs_measured.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Constants
NUM_TASKS_TO_DISPLAY = 20  # Number of tasks to display
Y_AXIS_MIN = 0  # Minimum value for Y-axis
Y_AXIS_MAX = 45  # Maximum value for Y-axis
BIAS = -2  # Bias to add/subtract from predicted data only (positive = add, negative = subtract)

labels = [str(i) for i in range(1, NUM_TASKS_TO_DISPLAY + 1)]

# ────────────────────────────────────────────────────────────────────────────────
# Old data
# ────────────────────────────────────────────────────────────────────────────────

# measured = np.array(
#     [
#         5.34,
#         5.38,
#         4.23,
#         3.96,
#         7.67,
#         5.35,
#         6.99,
#         5.48,
#         5.86,
#         7.37,
#         8.38,
#         15.17,
#         33.44,
#         15.01,
#         14.12,
#         21.79,
#         22.17,
#         26.72,
#         30.19,
#         68.61,
#     ]
# )
# predicted = np.array(
#     [
#         7.65,
#         7.86,
#         7.86,
#         7.86,
#         9.95,
#         9.95,
#         9.95,
#         9.95,
#         9.95,
#         9.95,
#         11.95,
#         15.74,
#         19.39,
#         19.48,
#         20.00,
#         30.17,
#         30.17,
#         38.38,
#         38.81,
#         108.77,
#     ]
# )

# ────────────────────────────────────────────────────────────────────────────────
# New data
# ────────────────────────────────────────────────────────────────────────────────

# Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
# --------------------------------------------------------------------------------
# SCH-B1G3L1M4-G214-226f         :         7.12            7.83           -9.07%
# SCH-B1G4L1M3-G298-856b         :         5.54            8.67          -36.07%
# SCH-B1G4M4-G298-102c           :         7.51            8.67          -13.36%
# SCH-B1G3M5-G374-0e78           :         9.70            9.43           +2.82%
# SCH-G3M2B4-G081-799c           :         8.02            9.58          -16.27%
# SCH-G3B1M5-G185-7c71           :         8.29            9.58          -13.44%
# SCH-G3B1L1M4-G194-6e9d         :         6.67            9.58          -30.42%
# SCH-G3M1L1B4-G260-a55b         :         8.23            9.58          -14.05%
# SCH-G3M2L1B3-G278-d0eb         :         5.77            9.58          -39.80%
# SCH-G3M1B3L2-G321-9db6         :         6.01            9.58          -37.30%
# SCH-G3B2M4-G210-c640           :         8.24            9.74          -15.36%
# SCH-G2M2L1B4-G310-44fc         :         9.83           10.22           -3.83%
# SCH-G3M3B3-G380-1095           :         7.00           10.60          -33.93%
# SCH-G4B5-G049-d312             :        12.35           11.43           +8.01%
# SCH-G4L1B4-G360-1216           :         8.65           11.43          -24.36%
# SCH-G4L1M4-G379-673e           :         7.27           11.43          -36.40%
# SCH-G3M4L2-G303-1b48           :         7.37           12.45          -40.86%
# SCH-B2M7-G059-2519             :        17.37           20.24          -14.18%
# SCH-B4L5-G306-e401             :        35.15           34.32           +2.40%
# SCH-B9-G000-a938               :        42.99           42.20           +1.87%

measured = np.array(
    [
        7.12,
        5.54,
        7.51,
        9.70,
        8.02,
        8.29,
        6.67,
        8.23,
        5.77,
        6.01,
        8.24,
        9.83,
        7.00,
        12.35,
        8.65,
        7.27,
        7.37,
        17.37,
        35.15,
        42.99,
    ]
)

predicted = np.array(
    [
        7.83,
        8.67,
        8.67,
        9.43,
        9.58,
        9.58,
        9.58,
        9.58,
        9.58,
        9.58,
        9.74,
        10.22,
        10.60,
        11.43,
        11.43,
        11.43,
        12.45,
        20.24,
        34.32,
        42.20,
    ]
)


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
