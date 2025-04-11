import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define core types mapping
CORE_TYPES = {0: "little", 1: "medium", 2: "big"}

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "omp.txt")

with open(input_file, "r") as f:
    content = f.read()

# Split content by device sections
device_pattern = r"\[\d+/\d+\]\s+Processing device:\s+([^\n]+)"
device_sections = re.split(device_pattern, content)

# First element is empty if file starts with a device section
if device_sections[0].strip() == "":
    device_sections.pop(0)

# Process each device section
results = {}
for i in range(0, len(device_sections), 2):
    if i + 1 < len(device_sections):
        device_id = device_sections[i].strip()
        device_content = device_sections[i + 1]

        # Extract benchmark data
        pattern = r"BM_run_OMP_stage/(\d+)/(\d+)\s+(\d+\.?\d*) ms"
        matches = re.findall(pattern, device_content)

        # Create dataframe for this device
        data = []
        for match in matches:
            stage, core_type, time_ms = match
            data.append(
                {
                    "Stage": int(stage),
                    "Core_Type": int(core_type),
                    "Core_Type_Name": CORE_TYPES[int(core_type)],
                    "Time_ms": float(time_ms),
                }
            )

        df = pd.DataFrame(data)
        results[device_id] = df.sort_values(by=["Stage", "Core_Type"])

# Print results for each device
for device_id, df in results.items():
    print(f"Device: {device_id}")
    print("=" * 50)
    print(df.to_string(index=False))
    print("\n")

# Optional: Compare performance across core types
print("Performance Comparison by Core Type")
print("=" * 50)
for device_id, df in results.items():
    print(f"Device: {device_id}")

    # Group by stage and calculate the best core type for each stage
    for stage, stage_df in df.groupby("Stage"):
        fastest_core = stage_df.loc[stage_df["Time_ms"].idxmin()]
        print(
            f"Stage {stage}: Fastest on {fastest_core['Core_Type_Name']} cores ({fastest_core['Time_ms']} ms)"
        )

    print("\n")

#  plot the results
# Create plots directory if it doesn't exist
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Set style
plt.style.use("ggplot")
sns.set_palette("colorblind")

# Plot for each device
for device_id, df in results.items():
    # 1. Plot per stage comparison of core types
    plt.figure(figsize=(14, 8))

    # Use seaborn for better visualization
    ax = sns.barplot(x="Stage", y="Time_ms", hue="Core_Type_Name", data=df)

    # Customize the plot
    plt.title(
        f"OMP Performance by Stage and Core Type - Device: {device_id}", fontsize=16
    )
    plt.xlabel("Pipeline Stage", fontsize=14)
    plt.ylabel("Time (ms)", fontsize=14)
    plt.yscale("log")  # Log scale to better visualize the differences
    plt.grid(True, alpha=0.3)
    plt.legend(title="Core Type")

    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"omp_performance_{device_id}.png"), dpi=300)

    # 2. Create heatmap showing relative performance
    plt.figure(figsize=(12, 8))

    # Pivot the data for the heatmap
    heatmap_data = df.pivot(index="Stage", columns="Core_Type_Name", values="Time_ms")

    # Plot heatmap
    ax = sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu_r", linewidths=0.5
    )

    plt.title(f"OMP Stage Performance (ms) - Device: {device_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"omp_heatmap_{device_id}.png"), dpi=300)

    # 3. Plot speedup of medium/big cores over little cores
    plt.figure(figsize=(14, 8))

    # Calculate relative speedup compared to little cores
    speedup_df = df.pivot(index="Stage", columns="Core_Type_Name", values="Time_ms")
    for col in speedup_df.columns:
        if col != "little":
            speedup_df[f"{col}_speedup"] = speedup_df["little"] / speedup_df[col]

    # Only keep speedup columns
    speedup_cols = [col for col in speedup_df.columns if "speedup" in col]
    speedup_df = speedup_df[speedup_cols]

    # Reset index to make Stage a column again
    speedup_df = speedup_df.reset_index()

    # Melt for seaborn
    melted_df = pd.melt(
        speedup_df, id_vars=["Stage"], var_name="Core_Type", value_name="Speedup"
    )
    melted_df["Core_Type"] = melted_df["Core_Type"].str.replace("_speedup", "")

    # Plot
    ax = sns.barplot(x="Stage", y="Speedup", hue="Core_Type", data=melted_df)

    # Add line at y=1 (no speedup)
    plt.axhline(y=1, color="r", linestyle="--", alpha=0.7)

    # Customize
    plt.title(f"Speedup vs Little Cores - Device: {device_id}", fontsize=16)
    plt.xlabel("Pipeline Stage", fontsize=14)
    plt.ylabel("Speedup (X times faster)", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1fx", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"omp_speedup_{device_id}.png"), dpi=300)

print(f"\nPlots saved to: {plots_dir}")
