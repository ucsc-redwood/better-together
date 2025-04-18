#!/usr/bin/env python3
import argparse
import subprocess
import os
import re
import io
import time
import pandas as pd


def get_next_log_filename(folder):
    """
    Determine the next log file name based on existing logs in the folder.
    Uses a naming convention 'run_<number>.txt'.
    """
    files = os.listdir(folder)
    indices = []
    pattern = re.compile(r"run_(\d+)\.txt")
    for f in files:
        match = pattern.match(f)
        if match:
            indices.append(int(match.group(1)))
    next_index = max(indices) + 1 if indices else 1
    return os.path.join(folder, f"run_{next_index}.txt")


def run_command(command):
    """
    Execute the given shell command and stream its output to the console.
    Also accumulate and return the complete output as a string.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    output_lines = []
    for line in process.stdout:
        print(line, end="")  # Show live progress on the console.
        output_lines.append(line)
    process.stdout.close()
    process.wait()
    return "".join(output_lines)


def parse_device_blocks(output):
    """
    Splits the output by device blocks. For each device, we assume that
    a line like "[... ] Processing device: <device_id>" appears. Then we look
    for the block starting at "### PYTHON_DATA_START ###" and ending at "### PYTHON_DATA_END ###".
    Returns a list of tuples: (device, block_text).
    """
    lines = output.splitlines()
    device = None
    blocks = []
    capture = False
    current_block = []
    for line in lines:
        # Try to detect a device marker.
        dev_match = re.search(r"Processing device:\s*(\S+)", line)
        if dev_match:
            # Update device id if a new block begins.
            device = dev_match.group(1)
            # If we were in the midst of capturing a block, finish it.
            if capture and current_block:
                blocks.append((device, "\n".join(current_block)))
                current_block = []
            capture = False

        # Start capturing when we see the PYTHON data marker.
        if "### PYTHON_DATA_START ###" in line:
            capture = True
            current_block = [line]
        elif "### PYTHON_DATA_END ###" in line and capture:
            current_block.append(line)
            blocks.append((device, "\n".join(current_block)))
            capture = False
            current_block = []
        elif capture:
            current_block.append(line)
    return blocks


def parse_benchmark_data(block_text):
    """
    Given the text block between the PYTHON data markers, extract both benchmark tables.
    The data block is expected to contain two CSV tables:
     - A table following the line "# NORMAL_BENCHMARK_DATA"
     - A table following the line "# FULLY_BENCHMARK_DATA"
    Returns a tuple of two pandas DataFrames.
    """
    # Extract only the section between the markers.
    start_idx = block_text.find("### PYTHON_DATA_START ###")
    end_idx = block_text.find("### PYTHON_DATA_END ###")
    if start_idx == -1 or end_idx == -1:
        return None
    data_text = block_text[start_idx:end_idx]

    # Find the markers for each data section.
    normal_marker = "# NORMAL_BENCHMARK_DATA"
    fully_marker = "# FULLY_BENCHMARK_DATA"
    normal_idx = data_text.find(normal_marker)
    fully_idx = data_text.find(fully_marker)

    if normal_idx == -1 or fully_idx == -1:
        return None

    # Extract the CSV text for the normal benchmark (from after its marker up to the fully marker)
    normal_block = data_text[normal_idx + len(normal_marker) : fully_idx].strip()
    # The fully benchmark table follows after its marker.
    fully_block = data_text[fully_idx + len(fully_marker) :].strip()

    try:
        df_normal = pd.read_csv(io.StringIO(normal_block))
        df_fully = pd.read_csv(io.StringIO(fully_block))
    except Exception as e:
        print("Error parsing CSV data: ", e)
        return None
    return df_normal, df_fully


def append_or_create_csv(df, file_path):
    """
    Appends the DataFrame to an existing CSV file if it exists;
    otherwise creates a new file with the header.
    """
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)


# Example usage
# py scripts-v2/collect/bm.py --log_folder data/2025-4-15/cifar-sparse/ --target bm-fully-cifar-sparse-vk --repeat 3
def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark, parse output, and store results."
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help="Folder path to store the output logs and accumulated CSV files",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=True,
        help="Number of times to run the benchmark command",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the benchmark target to run",
    )
    args = parser.parse_args()

    # Ensure the log folder exists.
    os.makedirs(args.log_folder, exist_ok=True)

    # The benchmark command to run.
    command = f"xmake r {args.target} -l off -t 4 -p"

    for run_num in range(1, args.repeat + 1):
        print(f"\n=== Run {run_num} of {args.repeat} ===")
        # Run the benchmark command and display output live.
        output = run_command(command)

        # Save the complete log output.
        log_file = get_next_log_filename(args.log_folder)
        with open(log_file, "w") as f:
            f.write(output)
        print(f"Saved log output to: {log_file}")

        # Parse the output for each device block.
        device_blocks = parse_device_blocks(output)
        if not device_blocks:
            print("No device data blocks found in the output.")
        for device, block in device_blocks:
            res = parse_benchmark_data(block)
            if res is None:
                print(f"No benchmark data found for device {device} in run {run_num}")
                continue
            df_normal, df_fully = res

            # For traceability, add device and run columns.
            df_normal["device"] = device
            df_normal["run"] = run_num
            df_fully["device"] = device
            df_fully["run"] = run_num

            # Define file paths based on the device id.
            normal_csv_path = os.path.join(args.log_folder, f"{device}_normal.csv")
            fully_csv_path = os.path.join(args.log_folder, f"{device}_fully.csv")

            # Append the new data to the CSV files.
            append_or_create_csv(df_normal, normal_csv_path)
            append_or_create_csv(df_fully, fully_csv_path)
            print(f"Appended benchmark data for device {device} in run {run_num}")

        # Optional: wait a bit between runs.
        time.sleep(1)


if __name__ == "__main__":
    main()
