import matplotlib.pyplot as plt
import numpy as np

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

# Simplify labels to show only the center part
labels = []
for label in original_labels:
    parts = label.split('-')
    if len(parts) >= 3:
        center_part = parts[1]
        hash_part = parts[-1]
        labels.append(f"{center_part}")
    else:
        labels.append(label)

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
plt.rcParams.update({'font.size': 12})  # Base font size for most elements

# Plot
x = np.arange(len(labels))
plt.figure(figsize=(14, 7))  # Slightly increase height for better label spacing
plt.plot(x, predicted, "r--", marker="s", markersize=8, label="Predicted", linewidth=2)
plt.errorbar(x, measured, yerr=0.5, fmt="b-", marker="^", markersize=8, label="Measured (Arithmetic)", linewidth=2)

plt.xticks(x, labels, rotation=45, ha='right', fontsize=12)  # Keep schedule UIDs at regular size
plt.ylabel("Time (Execution ms)", fontsize=16)
plt.xlabel("Execution Schedule", fontsize=16)
plt.title("Comparison of Measured vs Predicted Execution Times", fontsize=22)  # Bigger title
plt.legend(fontsize=18)  # Bigger legend font
plt.grid(True)
plt.tight_layout(pad=0.8)  # Reduce padding to minimize white space
plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin for x-axis labels
plt.savefig("example_predition", dpi=300, bbox_inches='tight')  # Higher DPI and tight bounding box
