import matplotlib.pyplot as plt
import numpy as np
import os

# Extracting the relevant data
original_labels = [
    "SCH-B1G3L1M4-G242",
    "SCH-M1G3L1B4-G197",
    "SCH-M1G3B3L2-G221",
    "SCH-B1G3M3L2-G264",
    "SCH-G3M2B4-G208",
    "SCH-G3B2M4-G277",
    "SCH-G3B1M5-G281",
    "SCH-G3B1L1M4-G281",
    "SCH-G3M1L1B4-G300",
    "SCH-G3M1B5-G300",
    "SCH-G4B5-G223",
    "SCH-G6L3-G046",
    "SCH-G9-G000",
    "SCH-B2M7-G111",
    "SCH-M2B7-G067",
    "SCH-M4L5-G029",
    "SCH-B4L5-G152",
    "SCH-B9-G000",
    "SCH-M9-G000",
    "SCH-L9-G000",
]

# # Simplify labels to show only the center part
# labels = []
# for label in original_labels:
#     parts = label.split("-")
#     if len(parts) >= 3:
#         center_part = parts[1]
#         hash_part = parts[-1]
#         labels.append(f"{center_part}")
#     else:
#         labels.append(label)

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
    label="Measured (Arithmetic)",
    linewidth=2,
)

plt.xticks(x, labels, ha="right", fontsize=22)
plt.ylabel("Time (Execution ms)", fontsize=24)
plt.yticks(fontsize=22)  # Set y-axis tick label font size
plt.legend(fontsize=24)
plt.grid(True)
plt.tight_layout(pad=0.8)
plt.subplots_adjust(bottom=0.2)


base_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(
    os.path.join(base_dir, "example_predition.svg"), bbox_inches="tight"
)
