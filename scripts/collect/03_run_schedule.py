#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run xmake benchmarks repeatedly and capture logs."
    )
    # Output folder
    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help="Root folder path for logs and CSV outputs",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to run the benchmark command",
    )

    # Basic target information
    parser.add_argument(
        "--app",
        type=str,
        choices=["cifar-sparse", "cifar-dense", "tree"],
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
        "--n-schedules-to-run",
        type=int,
        default=20,
        help="Number of schedules to run",
    )

    # The url has the following structure:
    #
    # http://192.168.1.12:8080/
    # ├── 3A021JEHN02756
    # │   ├── cifar-dense
    # │   │   └── vk
    # │   │       ├── schedules_btpm_gapness.json
    # │   │       ├── schedules_btpm_tmax.json
    # │   │       ├── schedules_isolated_gapness.json
    # │   │       └── schedules_isolated_tmax.json
    #
    parser.add_argument(
        "--schedules-server",
        type=str,
        default="http://192.168.1.12:8080",
        help="URL of the server hosting schedule JSON files",
    )

    # which table and optimization mode to use
    #
    parser.add_argument(
        "--table_type",
        type=str,
        choices=["isolated", "btpm"],
        required=True,
        help="Mode to select CSV file: 'isolated' for isolated.csv or 'btpm' for btpm.csv",
    )
    parser.add_argument(
        "--minimize_mode",
        type=str,
        choices=["gapness", "tmax"],
        required=True,
        help="Mode to minimize: 'gapness' for minimizing the gap between max and min chunk times, 'tmax' for minimizing the max chunk time",
    )

    # verbose
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose mode",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    log_folder = args.log_folder
    backend = args.backend
    device = args.device
    app = args.app
    table_type = args.table_type
    minimize_mode = args.minimize_mode
    repeat = args.repeat
    n_schedules_to_run = args.n_schedules_to_run
    schedules_server = args.schedules_server
    # verbose = args.verbose

    # Create the directory path with new structure
    log_path = os.path.join(log_folder, device, app, backend)
    os.makedirs(log_path, exist_ok=True)

    schedule_url = (
        f"{schedules_server}/"
        f"{device}/{app}/{backend}/schedules_{table_type}_{minimize_mode}.json"
    )

    # Command base
    cmd_base = [
        "xmake",
        "r",
        f"bm-gen-logs-{app}-{backend}",
        "--device",
        device,
        "--schedule-url",
        schedule_url,
        "--n-schedules-to-run",
        str(n_schedules_to_run),
    ]

    print(f"====== Running {repeat} times with command: {cmd_base} ======")

    for i in range(repeat):
        # Create individual log filename for each run
        log_filename = f"schedule_run_{i+1}.log"
        log_path_file = os.path.join(log_path, log_filename)

        print(f"Starting run {i+1}/{repeat}...")

        with open(log_path_file, "w") as log_file:
            # Launch the subprocess
            proc = subprocess.Popen(
                cmd_base,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in proc.stdout:
                print(line, end="")  # to console
                log_file.write(line)  # to file

            proc.wait()

            if proc.returncode != 0:
                print(
                    f"Warning: run {i+1} exited with code {proc.returncode}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Run {i+1}/{args.repeat} completed successfully. Log saved to: {log_path_file}"
                )

    print(f"\nAll {args.repeat} runs complete. Log files saved in: {log_path}")
    print(f"You can now run parse_schedules_by_widest.py on the folder: {log_path}")


if __name__ == "__main__":
    main()
