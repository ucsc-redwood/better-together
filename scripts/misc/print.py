#!/usr/bin/env python3
import argparse
import subprocess
import re
import statistics
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Run Pearson correlation on schedule runs 1–5 and report the average."
    )
    parser.add_argument('device',
                        help="Device ID (e.g., 3A021JEHN02756)")
    parser.add_argument('app',
                        help="Application name (e.g., tree)")
    parser.add_argument('backend',
                        help="Backend name (e.g., vk, cu)")
    args = parser.parse_args()

    device = args.device
    app = args.app
    backend = args.backend
    values = []

    for i in range(1, 6):
        log_path = (
            f"data/exe_logs_isolated_tmax/{device}/{app}/{backend}/"
            f"schedule_run_{i}.log"
        )
        sched_path = (
            f"data/schedules/{device}/{app}/{backend}/"
            "schedules_isolated_tmax.json"
        )

        print(f"=== schedule_run_{i}.log ===")
        cmd = [
            "uv", "run", "scripts/collect/04_parse_schedules.py", "-v",
            log_path,
            "--schedule-file", sched_path,
            "--time-window", "0.25-0.75"
        ]
        try:
            proc = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  check=True)
        except subprocess.CalledProcessError as e:
            print(f"[error] run {i} failed: {e}", file=sys.stderr)
            continue

        # find and parse the coefficient
        for line in proc.stdout.splitlines():
            if "Pearson correlation coefficient" in line:
                # assume format "...: 0.9262"
                num_str = line.split(":")[-1].strip()
                try:
                    val = float(num_str)
                    print(val)
                    values.append(val)
                except ValueError:
                    print(f"[warn] couldn't parse value in line: {line}", file=sys.stderr)
                break

    if values:
        avg = statistics.mean(values)
        print(f"\nAverage Pearson correlation coefficient: {avg:.4f}")
    else:
        print("No valid coefficients extracted.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
