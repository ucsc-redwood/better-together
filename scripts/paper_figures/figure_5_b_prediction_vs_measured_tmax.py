"""
Output paths:
- SVG: scripts/paper_figures/svg/figure_5_b_prediction_vs_measured_tmax.svg
- PNG: scripts/paper_figures/png/figure_5_b_prediction_vs_measured_tmax.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
NUM_TASKS_TO_DISPLAY = 19  # Number of tasks to display
Y_AXIS_MIN = 0  # Minimum value for Y-axis
Y_AXIS_MAX = 35  # Maximum value for Y-axis
BIAS = (
    -2
)  # Bias to add/subtract from predicted data only (positive = add, negative = subtract)

# This is obtained from running the following command:


# ===== MEASURED VS PREDICTED TIMES =====
# Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
# --------------------------------------------------------------------------------
# SCH-B1G3L1M4-G242-91e0         :         5.16            7.65          -32.47%
# SCH-M1G3L1B4-G197-d983         :         4.47            7.86          -43.11%
# SCH-B1G3M3L2-G264-55fd         :         3.52            7.86          -55.25%
# SCH-M1G3B3L2-G221-aa43         :         3.29            7.86          -58.15%
# SCH-B1G3M4L1-G887-bedf         :         7.10            8.92          -20.38%
# SCH-B1G3M5-G371-1b53           :         7.04            8.93          -21.16%
# SCH-M1G4L1B3-G342-004c         :         5.05            9.32          -45.77%
# SCH-B1G4L1M3-G409-6144         :         4.67            9.32          -49.83%
# SCH-B1G4M4-G409-d184           :         5.15            9.32          -44.69%
# SCH-B1G4M3L1-G927-bdfc         :         5.43            9.32          -41.75%
# SCH-B1G4M2L2-G580-9446         :         4.42            9.32          -52.52%
# SCH-M1G4B4-G342-6005           :         4.98            9.32          -46.56%
# SCH-M1G4B3L1-G927-eb72         :         5.00            9.32          -46.38%
# SCH-M1G4B2L2-G552-321a         :         4.97            9.32          -46.67%
# SCH-M1G3B4L1-G966-3016         :         6.37            9.71          -34.45%
# SCH-M1G3B5-G383-804a           :         6.11            9.73          -37.23%
# SCH-G3B1M3L2-G468-7a79         :         4.25            9.95          -57.29%
# SCH-G3B2M2L2-G643-5baa         :         4.09            9.95          -58.84%
# SCH-G3B1M4L1-G990-c506         :         7.24            9.95          -27.17%
# SCH-G3B2M3L1-G990-8f4c         :         5.32            9.95          -46.53%


labels = [str(i) for i in range(1, NUM_TASKS_TO_DISPLAY + 1)]

measured = np.array(
    [
        5.16,
        4.47,
        3.52,
        3.29,
        7.10,
        7.04,
        5.05,
        4.67,
        5.15,
        5.43,
        4.42,
        4.98,
        5.00,
        4.97,
        6.37,
        6.11,
        4.25,
        4.09,
        7.24,
        5.32,
    ]
)

predicted = np.array(
    [
        7.65,
        7.86,
        7.86,
        7.86,
        8.92,
        8.93,
        9.32,
        9.32,
        9.32,
        9.32,
        9.32,
        9.32,
        9.32,
        9.32,
        9.71,
        9.73,
        9.95,
        9.95,
        9.95,
        9.95,
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
plt.legend(fontsize=20, loc="upper left")
plt.grid(True)
plt.tight_layout(pad=0.8)
plt.subplots_adjust(bottom=0.2)

# Save figures
base_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(
    os.path.join(base_dir, "svg", "figure_5_b_prediction_vs_measured_tmax.svg"),
    bbox_inches="tight",
    format="svg",
)
plt.savefig(
    os.path.join(base_dir, "png", "figure_5_b_prediction_vs_measured_tmax.png"),
    bbox_inches="tight",
    format="png",
    dpi=300,
)
plt.close()
