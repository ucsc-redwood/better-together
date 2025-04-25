#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run xmake benchmarks repeatedly and capture logs."
    )
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
    parser.add_argument(
        "--schedules-server",
        type=str,
        default="http://192.168.1.204:8080",
        help="URL of the server hosting schedule JSON files",
    )
    parser.add_argument(
        "--use-normal-table",
        type=bool,
        default=False,
        help="Use normal table for schedule",
    )
    args = parser.parse_args()

    # Create the directory path with new structure
    log_path = os.path.join(args.log_folder, args.device, args.app, args.backend)
    os.makedirs(log_path, exist_ok=True)

    # Base schedule URL
    if args.use_normal_table:
        schedule_url = (
            f"{args.schedules_server}/"
            f"{args.device}/{args.app}/{args.backend}/schedules_normal.json"
        )
    else:
        schedule_url = (
            f"{args.schedules_server}/"
            f"{args.device}/{args.app}/{args.backend}/schedules.json"
        )

    # Command base
    cmd_base = [
        "xmake",
        "r",
        f"bm-gen-logs-{args.app}-{args.backend}",
        "--device",
        args.device,
        "--schedule-url",
        schedule_url,
        "--n-schedules-to-run",
        str(args.n_schedules_to_run),
    ]

    print(f"====== Running {args.repeat} times with command: {cmd_base} ======")

    for i in range(args.repeat):
        # Create individual log filename for each run
        log_filename = f"schedule_run_{i+1}.log"
        log_path_file = os.path.join(log_path, log_filename)

        print(f"Starting run {i+1}/{args.repeat}...")

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


# Example usage:
# python3 scripts/collect/03_run_schedule.py \
#   --log_folder ./data/exe_logs \
#   --repeat 5 \
#   --app cifar-sparse \
#   --backend vk \
#   --device 3A021JEHN02756
if __name__ == "__main__":
    main()
