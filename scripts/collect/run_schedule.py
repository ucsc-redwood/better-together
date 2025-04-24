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
        "--result_folder",
        type=str,
        required=True,
        help="Folder path for logs and CSV outputs",
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
    args = parser.parse_args()

    # Prepare output folder
    os.makedirs(args.result_folder, exist_ok=True)

    # Log file for all runs
    log_filename = f"{args.device}_{args.app}_{args.backend}_pipeline_results.log"
    log_path = os.path.join(args.result_folder, log_filename)

    # Base schedule URL prefix
    schedule_url = (
        f"http://192.168.1.204:8080/"
        f"{args.device}_{args.app}_{args.backend}_fully_schedules.json"
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
        "10",
    ]

    with open(log_path, "w") as log_file:
        for i in range(args.repeat):
            header = f"\n=== Run {i+1}/{args.repeat} ===\n"
            print(header, end="")
            log_file.write(header)

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
            footer = f"--- Exit code: {proc.returncode} ---\n"
            print(footer, end="")
            log_file.write(footer)

            if proc.returncode != 0:
                print(
                    f"Warning: iteration {i+1} exited with code {proc.returncode}",
                    file=sys.stderr,
                )

    print(f"\nAll runs complete. Combined log at: {log_path}")


# python3 scripts/collect/run_schedule.py \
#   --result_folder ./logs \
#   --repeat 5 \
#   --app cifar-sparse \
#   --backend vk \
#   --device 3A021JEHN02756
if __name__ == "__main__":
    main()
