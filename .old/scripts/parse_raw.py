#!/usr/bin/env python3

import json
import sys
import argparse
from common import parse_run_name


def pritty_print_entry(entry):
    """Pretty print a benchmark entry showing just run_name and real_time."""
    run_name = parse_run_name(entry["run_name"])
    print(run_name)


def main():
    parser = argparse.ArgumentParser(
        description="Parse and display Google Benchmark JSON results"
    )
    parser.add_argument("json_file", help="Path to Google Benchmark JSON file")
    parser.add_argument("device_name", help="Device name to filter results")
    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        data = json.load(f)

    # Ensure the JSON has the "benchmarks" key
    if "benchmarks" not in data:
        print("Error: Invalid Google Benchmark JSON file (no 'benchmarks' key).")
        sys.exit(1)

    # Filter and print only those entries whose aggregate_name is "mean"
    for benchmark in data["benchmarks"]:
        if benchmark.get("aggregate_name") == "mean":
            pritty_print_entry(benchmark)


if __name__ == "__main__":
    main()
