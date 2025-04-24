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
    parser.add_argument(
        "--n-schedules-to-run",
        type=int,
        default=10,
        help="Number of schedules to run",
    )
    args = parser.parse_args()

    # Prepare output folder
    os.makedirs(args.result_folder, exist_ok=True)

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
        str(args.n_schedules_to_run),
    ]

    for i in range(args.repeat):
        # Create individual log filename for each run
        log_filename = f"{args.device}_{args.app}_{args.backend}_schedules_{i+1}.log"
        log_path = os.path.join(args.result_folder, log_filename)

        print(f"Starting run {i+1}/{args.repeat}...")

        with open(log_path, "w") as log_file:
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
                    f"Run {i+1}/{args.repeat} completed successfully. Log saved to: {log_path}"
                )

    print(
        f"\nAll {args.repeat} runs complete. Log files saved in: {args.result_folder}"
    )
    print(
        f"You can now run parse_schedules_by_widest.py on the folder: {args.result_folder}"
    )


# python3 scripts/collect/run_schedule.py \
#   --result_folder ./logs \
#   --repeat 5 \
#   --app cifar-sparse \
#   --backend vk \
#   --device 3A021JEHN02756
if __name__ == "__main__":
    main()
