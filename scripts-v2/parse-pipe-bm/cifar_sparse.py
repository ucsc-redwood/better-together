import pandas as pd
import re
import os

# assume same directory

base_dir = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(base_dir, "cifar_sparse.txt"), "r") as f:
    log_text = f.read()

# Split the log text into individual lines
lines = log_text.strip().splitlines()

# Create an empty list to hold parsed rows
rows = []

# Define a regex pattern to split columns (assuming two or more whitespace characters separate columns)
pattern = re.compile(r"\s{2,}")

# Iterate over each line and filter only rows that contain "_mean"
for line in lines:
    # Skip header/dash lines (and any empty lines)
    if "_mean" not in line:
        continue

    # Split the line into columns
    parts = pattern.split(line.strip())
    if len(parts) < 2:
        continue

    benchmark_label = parts[
        0
    ]  # e.g. "VK/CifarSparse/Stage/1/iterations:1/repeats:5_mean"
    time_value = parts[1]  # e.g. "512 ms" or "607 ms" etc.

    # Remove the "ms" and extra spaces, then convert to float.
    time_numeric = float(time_value.replace("ms", "").strip())

    # Initialize the extracted fields
    vendor = None
    run_type = None  # "Baseline" or "Stage"
    stage = None
    core = None

    # Split the benchmark label by '/'.
    bench_parts = benchmark_label.split("/")
    # The first part is the vendor
    vendor = bench_parts[0]

    # Now check what type of result this is (Baseline or Stage)
    if "Baseline" in bench_parts:
        run_type = "Baseline"
        stage = None  # No stage for baseline
    elif "Stage" in bench_parts:
        run_type = "Stage"
        # Expect structure like: Vendor, CifarSparse, Stage, <stage>, (possibly <core>), then other parts
        try:
            # Find the index where "Stage" occurs.
            stage_index = bench_parts.index("Stage")
            # The next element should be the stage number.
            stage = bench_parts[stage_index + 1]
            # For OMP benchmarks, there might be an extra numeric component
            # Check if there is one more part that is not the iterations info.
            if len(bench_parts) > stage_index + 2:
                # If the next part is a digit (or represents a core type) then capture it.
                potential_core = bench_parts[stage_index + 2]
                if re.match(r"^\d+$", potential_core):
                    core = potential_core
        except (IndexError, ValueError):
            stage = None

    # Append the parsed data as a dictionary
    rows.append(
        {
            "Vendor": vendor,
            "Type": run_type,
            "Stage": stage,
            "Core": core,
            "MeanTime_ms": time_numeric,
        }
    )

# Create a pandas DataFrame from the parsed rows
df = pd.DataFrame(rows)

# Print the extracted DataFrame
print(df)
