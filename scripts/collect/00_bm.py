#!/usr/bin/env python3
import argparse
import subprocess
import os
import re
import io
import time
import glob
import pandas as pd


def get_next_log_filename(log_path):
    """
    Determine the next log file name based on existing logs in the folder.
    Uses naming convention: '<run_id>.log'
    """
    files = os.listdir(log_path)
    indices = []
    pattern = re.compile(r"(\d+)\.log")
    for f in files:
        match = pattern.match(f)
        if match:
            indices.append(int(match.group(1)))
    next_index = max(indices) + 1 if indices else 1
    filename = f"{next_index}.log"
    return os.path.join(log_path, filename)


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


def aggregate_existing_logs(folder_root, device, app, backend):
    """
    Parse all existing logs for the given device/app/backend
    and append their data to the appropriate CSV files.
    """
    log_path = os.path.join(folder_root, device, app, backend)
    pattern = os.path.join(log_path, "*.log")
    log_files = sorted(glob.glob(pattern))
    normal_csv = os.path.join(log_path, "normal.csv")
    fully_csv = os.path.join(log_path, "fully.csv")

    for log_file in log_files:
        # extract run number from filename
        match = re.search(r"(\d+)\.log$", log_file)
        run_num = int(match.group(1)) if match else None

        with open(log_file) as f:
            output = f.read()
        parsed = parse_benchmark_data(output)
        if not parsed:
            print(f"No benchmark data in {log_file}")
            continue
        df_normal, df_fully = parsed
        df_normal["device"] = device
        df_fully["device"] = device
        if run_num is not None:
            df_normal["run"] = run_num
            df_fully["run"] = run_num
        append_or_create_csv(df_normal, normal_csv)
        append_or_create_csv(df_fully, fully_csv)
        print(f"Aggregated data from {os.path.basename(log_file)}")
    print(f"Finished aggregating logs into {normal_csv} and {fully_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run or aggregate benchmarks.")
    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help="Root folder path for logs and CSV outputs",
    )
    parser.add_argument(
        "--repeat",
        type=int,
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
        help="Device identifier to deploy/run or aggregate",
    )
    parser.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Only aggregate existing logs; do not run benchmarks",
    )
    args = parser.parse_args()

    folder_root = args.log_folder
    device = args.device
    app = args.app
    backend = args.backend
    target = f"bm-fully-{app}-{backend}"

    # Create directory structure
    log_path = os.path.join(folder_root, device, app, backend)
    os.makedirs(log_path, exist_ok=True)

    if args.only_aggregate:
        aggregate_existing_logs(folder_root, device, app, backend)
        return

    # Otherwise, run benchmarks as usual
    if args.repeat is None:
        parser.error("--repeat is required when not using --only-aggregate")
    command_template = f"xmake r {target} --device {device} -l off -p -t 1"

    for run_num in range(1, args.repeat + 1):
        print(f"\n=== Run {run_num} of {args.repeat} ===")
        output = run_command(command_template)
        log_file = get_next_log_filename(log_path)
        with open(log_file, "w") as f:
            f.write(output)
        print(f"Saved log output to: {log_file}")
        parsed = parse_benchmark_data(output)
        if not parsed:
            print("No benchmark data found in the output.")
            continue
        df_normal, df_fully = parsed
        df_normal["device"] = device
        df_normal["run"] = run_num
        df_fully["device"] = device
        df_fully["run"] = run_num
        normal_csv = os.path.join(log_path, "normal.csv")
        fully_csv = os.path.join(log_path, "fully.csv")
        append_or_create_csv(df_normal, normal_csv)
        append_or_create_csv(df_fully, fully_csv)
        print(
            f"Appended benchmark data for device {device}, app {app},"
            f" backend {backend}, run {run_num}"
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
