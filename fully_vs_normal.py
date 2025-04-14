#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style (optional)
sns.set(style="whitegrid")

# Assumed stage names (one per block; file is expected to have 9 blocks)
STAGE_NAMES = [f"Stage {i}" for i in range(1, 10)]


def parse_benchmark_file(filepath, expected_processors):
    """
    Parse the benchmark file and return a list of lists.
    Each inner list corresponds to one stage, containing float AVG values
    for the processors in the order defined by the device's processors list.

    Expected file format (each block separated by a blank line):
      PROCESSOR=Little|AVG=19.0214
      PROCESSOR=Medium|AVG=7.01616
      PROCESSOR=Big|AVG=5.05856
      PROCESSOR=GPU|AVG=6.0146

      PROCESSOR=Little|AVG=22.8604
      PROCESSOR=Medium|AVG=12.6958
      ... etc.
    """
    stages = []
    current_block = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                current_block.append(line)
            else:
                if current_block:
                    stages.append(current_block)
                    current_block = []
        # In case the file does not end with a blank line:
        if current_block:
            stages.append(current_block)

    # Parse each block to get the AVG values
    data = []
    for block in stages:
        if len(block) != len(expected_processors):
            raise ValueError(
                f"Expected {len(expected_processors)} lines per stage, got {len(block)} in block: {block}"
            )
        stage_values = []
        for line in block:
            # Expected format: "PROCESSOR=Name|AVG=Value"
            try:
                parts = line.split("|")
                avg_part = parts[1]
                value = float(avg_part.split("=")[1])
            except Exception as e:
                raise ValueError(f"Error parsing line: {line}\n{e}")
            stage_values.append(value)
        data.append(stage_values)
    return data


def create_dataframe(data, device_name, condition, processors):
    """
    Convert parsed data into a Pandas DataFrame.
    'data' is a list of lists;
    'condition' is a string, e.g., 'Normal' or 'Occupied'.
    """
    df = pd.DataFrame(data, columns=processors, index=STAGE_NAMES)
    df.index.name = f"Stage ({condition})"
    return df


def process_device(normal_file, occupied_file, device_id, processors):
    print(f"Processing device {device_id} ...")

    # Parse the input files
    normal_data = parse_benchmark_file(normal_file, processors)
    occupied_data = parse_benchmark_file(occupied_file, processors)

    # Create DataFrames for normal and occupied conditions
    df_normal = create_dataframe(normal_data, device_id, "Normal", processors)
    df_occupied = create_dataframe(occupied_data, device_id, "Occupied", processors)

    # Calculate ratio and difference DataFrames
    df_ratio = df_occupied / df_normal
    df_diff = df_occupied - df_normal

    # Compute average ratio and difference per processor (columns)
    mean_ratio = df_ratio.mean()
    mean_diff = df_diff.mean()

    # --- Plotting ---
    output_prefix = f"{device_id}"

    # 1. Heatmap for Ratio (Occupied/Normal)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_ratio, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Ratio (Occupied / Normal) Per Stage - Device {device_id}")
    plt.ylabel("Stage")
    plt.xlabel("Processor")
    plt.tight_layout()
    ratio_heatmap_file = f"{output_prefix}_ratio_heatmap.png"
    plt.savefig(ratio_heatmap_file)
    print(f"Saved {ratio_heatmap_file}")
    plt.close()

    # 2. Heatmap for Difference (Occupied - Normal)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_diff, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Difference (Occupied - Normal) Per Stage (ms) - Device {device_id}")
    plt.ylabel("Stage")
    plt.xlabel("Processor")
    plt.tight_layout()
    diff_heatmap_file = f"{output_prefix}_diff_heatmap.png"
    plt.savefig(diff_heatmap_file)
    print(f"Saved {diff_heatmap_file}")
    plt.close()

    # 3. Bar Chart for Average Ratio Per Processor
    plt.figure(figsize=(8, 4))
    mean_ratio.plot(kind="bar")
    plt.title(f"Average Ratio Per Processor (Occupied / Normal) - Device {device_id}")
    plt.ylabel("Ratio")
    plt.xlabel("Processor")
    plt.tight_layout()
    avg_ratio_bar_file = f"{output_prefix}_avg_ratio.png"
    plt.savefig(avg_ratio_bar_file)
    print(f"Saved {avg_ratio_bar_file}")
    plt.close()

    # 4. Bar Chart for Average Difference Per Processor
    plt.figure(figsize=(8, 4))
    mean_diff.plot(kind="bar", color="orange")
    plt.title(f"Average Difference Per Processor (ms) - Device {device_id}")
    plt.ylabel("Difference (ms)")
    plt.xlabel("Processor")
    plt.tight_layout()
    avg_diff_bar_file = f"{output_prefix}_avg_diff.png"
    plt.savefig(avg_diff_bar_file)
    print(f"Saved {avg_diff_bar_file}")
    plt.close()


def main():
    # Hardcode file paths (adjust if needed)
    devices = {
        "3A021JEHN02756": {
            "normal": "BM_table_cifar_sparse_vk_3A021JEHN02756.txt",
            "occupied": "BM_table_cifar_sparse_vk_3A021JEHN02756_full.txt",
            "processors": ["Little", "Medium", "Big", "GPU"],
        },
        "9b034f1b": {
            "normal": "BM_table_cifar_sparse_vk_9b034f1b.txt",
            "occupied": "BM_table_cifar_sparse_vk_9b034f1b_full.txt",
            "processors": ["Little", "Medium", "Big", "GPU"],
        },
        "jetson": {
            "normal": "BM_table_cifar_sparse_vk_jetson.txt",
            "occupied": "BM_table_cifar_sparse_vk_jetson_full.txt",
            "processors": ["Little", "GPU"],
        },
    }

    # Process each device
    for device_id, paths in devices.items():
        normal_file = paths["normal"]
        occupied_file = paths["occupied"]
        processors = paths["processors"]

        if not os.path.exists(normal_file):
            print(f"File not found: {normal_file}")
            continue
        if not os.path.exists(occupied_file):
            print(f"File not found: {occupied_file}")
            continue

        process_device(normal_file, occupied_file, device_id, processors)


if __name__ == "__main__":
    main()
