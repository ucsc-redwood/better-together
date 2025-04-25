import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np

data = """
stage,little,medium,big,vulkan,cuda,device,run
1,2.6552,0.8293,0.6454,0.8391,0.0,3A021JEHN02756,1
2,3.3579,1.5446,1.4765,21.3617,0.0,3A021JEHN02756,1
3,3.1741,0.327,0.3119,1.4227,0.0,3A021JEHN02756,1
4,9.5524,5.4216,5.2513,1.3518,0.0,3A021JEHN02756,1
5,1.4963,0.2796,0.2337,0.607,0.0,3A021JEHN02756,1
6,1.5167,0.3745,0.391,0.6942,0.0,3A021JEHN02756,1
7,5.6034,2.0263,2.1205,1.5197,0.0,3A021JEHN02756,1
"""

df = pd.read_csv(StringIO(data))

# Select tasks that show different optimal processors
tasks = df[df["stage"].isin([2, 4, 7])]

# Set style
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Create figure and axis
plt.figure(figsize=(6, 4))

# Set up the bar positions
x = np.arange(3)  # 2 tasks
width = 0.2  # width of the bars

# Plot all processor types
plt.bar(x - width * 1.5, tasks["little"], width, label="CPU (Little)", color="#1f77b4")
plt.bar(x - width / 2, tasks["medium"], width, label="CPU (Medium)", color="#2ca02c")
plt.bar(x + width / 2, tasks["big"], width, label="CPU (Big)", color="#ff7f0e")
plt.bar(x + width * 1.5, tasks["vulkan"], width, label="GPU", color="#d62728")

# Add labels and title
plt.xlabel("Task")
plt.ylabel("Execution Time (ms)")
plt.title("Processor Performance Comparison")
plt.xticks(x, ["Sort", "Build Radix Tree", "Build Octree"])
plt.legend(loc="upper right")

# Add value labels on top of bars
for i, (l, m, b, g) in enumerate(
    zip(tasks["little"], tasks["medium"], tasks["big"], tasks["vulkan"])
):
    plt.text(i - width * 1.5, l + 0.1, f"{l:.2f}", ha="center", fontsize=8)
    plt.text(i - width / 2, m + 0.1, f"{m:.2f}", ha="center", fontsize=8)
    plt.text(i + width / 2, b + 0.1, f"{b:.2f}", ha="center", fontsize=8)
    plt.text(i + width * 1.5, g + 0.1, f"{g:.2f}", ha="center", fontsize=8)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("processor_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
