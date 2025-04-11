import pandas as pd
import re
import os

# assume same directory

base_dir = os.path.dirname(os.path.abspath(__file__))


def parse_benchmark_line(line):
    """Parse a single benchmark line and extract the data."""
    # Define a regex pattern to split columns (assuming two or more whitespace characters separate columns)
    pattern = re.compile(r"\s{2,}")

    # Skip header/dash lines (and any empty lines)
    if "_mean" not in line:
        return None

    # Split the line into columns
    parts = pattern.split(line.strip())
    if len(parts) < 2:
        return None

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

    # Return the parsed data as a dictionary
    return {
        "Vendor": vendor,
        "Type": run_type,
        "Stage": stage,
        "Core": core,
        "MeanTime_ms": time_numeric,
    }


def extract_benchmark_data(log_text):
    """Extract benchmark data from the log text."""
    # Split the log text into individual lines
    lines = log_text.strip().splitlines()

    # Create an empty list to hold parsed rows
    rows = []

    # Parse each line
    for line in lines:
        result = parse_benchmark_line(line)
        if result:
            rows.append(result)

    # Create a pandas DataFrame from the parsed rows
    return pd.DataFrame(rows)


def split_log_by_devices(log_text):
    """Split the log text into chunks by device."""
    device_pattern = re.compile(r"\[\d+/\d+\]\s+Processing device:\s+([^\n]+)")
    device_matches = list(device_pattern.finditer(log_text))

    device_chunks = {}

    # For each device found
    for i, match in enumerate(device_matches):
        device_id = match.group(1).strip()
        start_pos = match.start()

        # Find the end of this chunk (start of next chunk or end of text)
        if i < len(device_matches) - 1:
            end_pos = device_matches[i + 1].start()
        else:
            end_pos = len(log_text)

        # Extract the chunk for this device
        chunk = log_text[start_pos:end_pos]
        device_chunks[device_id] = chunk

    return device_chunks


# Read the log file
with open(os.path.join(base_dir, "cifar_sparse.txt"), "r") as f:
    log_text = f.read()

# Split the log text by devices
device_chunks = split_log_by_devices(log_text)

# Process each device chunk separately
device_dfs = {}
for device_id, chunk in device_chunks.items():
    print(f"Processing data for device: {device_id}")
    df = extract_benchmark_data(chunk)
    device_dfs[device_id] = df
    print(f"Device {device_id} DataFrame:")
    print(df)
    print("\n")

# Access each device's DataFrame as needed
# device_dfs['3A021JEHN02756'] for first device
# device_dfs['9b034f1b'] for second device
