#!/usr/bin/env python3
import argparse
import subprocess
import os
import re
import io
import time
import pandas as pd


def get_next_log_filename(folder, device, app, backend):
    """
    Determine the next log file name based on existing logs in the folder
    for a given device, app, backend. Uses naming convention:
    'bm_<device>_<app>_<backend>_<number>.log'
    """
    files = os.listdir(folder)
    indices = []
    pattern = re.compile(
        rf"bm_{re.escape(device)}_{re.escape(app)}_{re.escape(backend)}_(\d+)\.log"
    )
    for f in files:
        match = pattern.match(f)
        if match:
            indices.append(int(match.group(1)))
    next_index = max(indices) + 1 if indices else 1
    filename = f"bm_{device}_{app}_{backend}_{next_index}.log"
    return os.path.join(folder, filename)


def run_command(command):
    """Execute the given shell command and return its full output."""
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
        print(line, end="")
        output_lines.append(line)
    process.stdout.close()
    retcode = process.wait()
    if retcode != 0:
        raise RuntimeError(f"Command failed with exit code {retcode}")
    return "".join(output_lines)


def parse_benchmark_data(output):
    """
    Given full output containing PYTHON_DATA_START/END markers,
    extract two pandas DataFrames for normal and fully benchmarks.
    """
    start_marker = "### PYTHON_DATA_START ###"
    end_marker = "### PYTHON_DATA_END ###"
    start_idx = output.find(start_marker)
    end_idx = output.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        return None
    data_text = output[start_idx:end_idx]
    normal_marker = "# NORMAL_BENCHMARK_DATA"
    fully_marker = "# FULLY_BENCHMARK_DATA"
    normal_idx = data_text.find(normal_marker)
    fully_idx = data_text.find(fully_marker)
    if normal_idx == -1 or fully_idx == -1:
        return None
    normal_block = data_text[normal_idx + len(normal_marker) : fully_idx].strip()
    fully_block = data_text[fully_idx + len(fully_marker) :].strip()
    if not normal_block or not fully_block:
        return None
    try:
        df_normal = pd.read_csv(io.StringIO(normal_block))
        df_fully = pd.read_csv(io.StringIO(fully_block))
    except Exception as e:
        print("Error parsing CSV data:", e)
        return None
    return df_normal, df_fully


def append_or_create_csv(df, file_path):
    """Append DataFrame to CSV or create file if not exists."""
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)


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
        "--app",
        type=str,
        required=True,
        help="Application name (e.g., cifar-sparse)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vk", "cu"],
        required=True,
        help="Backend type: 'vk' or 'cu'",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device identifier to deploy and run the benchmark on",
    )
    args = parser.parse_args()

    # Construct target and naming
    app = args.app
    backend = args.backend
    device = args.device
    target = f"bm-fully-{app}-{backend}"

    os.makedirs(args.log_folder, exist_ok=True)
    command_template = f"xmake r {target} --device {device} -l off -p -t 5"

    for run_num in range(1, args.repeat + 1):
        print(f"\n=== Run {run_num} of {args.repeat} ===")
        output = run_command(command_template)

        # Save log with device/app/backend naming
        log_file = get_next_log_filename(args.log_folder, device, app, backend)
        with open(log_file, "w") as f:
            f.write(output)
        print(f"Saved log output to: {log_file}")

        # Parse embedded Python benchmark data
        parsed = parse_benchmark_data(output)
        if not parsed:
            print("No benchmark data found in the output.")
            continue
        df_normal, df_fully = parsed
        df_normal["device"] = device
        df_normal["run"] = run_num
        df_fully["device"] = device
        df_fully["run"] = run_num

        # Append to per-device CSVs
        normal_csv = os.path.join(args.log_folder, f"{device}_normal.csv")
        fully_csv = os.path.join(args.log_folder, f"{device}_fully.csv")
        append_or_create_csv(df_normal, normal_csv)
        append_or_create_csv(df_fully, fully_csv)
        print(f"Appended benchmark data for device {device} in run {run_num}")

        time.sleep(1)


if __name__ == "__main__":
    main()
