import argparse
import subprocess
import sys
import os
from typing import List, Dict, Tuple

# hardcoded paths
RAW_BENCHMARK_PATH = "data/raw_bm_results"
RAW_LOGS_PATH = "data/logs"
GENERATED_SCHEDULES_PATH = "data/generated-schedules"

DB_PATH = "data/benchmark_results.db"

# Constants moved to top level
ALL_BENCHMARKS = [
    "bm-cifar-dense-omp",
    "bm-cifar-dense-vk",
    "bm-cifar-sparse-omp",
    "bm-cifar-sparse-vk",
    "bm-tree-omp",
    "bm-tree-vk",
]

ALL_DEVICES = ["3A021JEHN02756", "9b034f1b", "ce0717178d7758b00b7e"]
ALL_APPLICATIONS = ["tree", "cifar-dense", "cifar-sparse"]

# Map application names to their canonical form
APPLICATION_NAME_MAP: Dict[str, str] = {
    "tree": "Tree",
    "cifar-sparse": "CifarSparse",
    "cifar-dense": "CifarDense",
}


def run_command(cmd: str, hide_output: bool = False) -> None:
    """Run a shell command and exit if it fails.

    Args:
        cmd: Command string to execute
        hide_output: If True, suppress command output
    """
    print(f"Executing: {cmd}")
    try:
        if hide_output:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: Command failed: {cmd}", file=sys.stderr)
        sys.exit(1)


def interactive_select(options: List[str], option_type: str) -> str:
    """
    Prompt the user to select a single option by number.

    Args:
        options: List of options to choose from
        option_type: Type of option (e.g. "device" or "application")

    Returns:
        Selected option
    """
    print(f"\nSelect {option_type} to use:")
    for i, option in enumerate(options):
        print(f"{i+1}: {option}")

    while True:
        selection = input("Enter your choice (1-{}): ".format(len(options))).strip()
        if not selection:
            print(f"Using first {option_type}: {options[0]}")
            return options[0]

        try:
            index = int(selection) - 1
            if 0 <= index < len(options):
                return options[index]
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def parse_schedule_range(range_str: str) -> set:
    """Parse schedule range string into a set of schedule IDs.

    Accepts formats:
    - Single number: "1"
    - Comma-separated: "1,3,5"
    - Range: "1-5"
    - Mixed: "1-3,5,7-9"
    """
    if not range_str:
        return set()

    schedule_ids = set()
    parts = range_str.split(",")

    for part in parts:
        if "-" in part:
            # Handle range
            try:
                start, end = map(int, part.split("-"))
                schedule_ids.update(range(start, end + 1))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid range format: {part}. Expected format: start-end"
                )
        else:
            # Handle single number
            try:
                schedule_ids.add(int(part))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid schedule ID: {part}. Must be an integer"
                )

    return schedule_ids


def get_num_schedules(device: str, application_name: str) -> int:
    """Get the number of schedules for a given device and application.

    By looking at the files in the `GENERATED_SCHEDULES_PATH` directory.
    We expect the files to be in the format `<device>_<application>_schedule_<id>.json`

    """
    schedules_dir = GENERATED_SCHEDULES_PATH
    num_schedules = 0

    for each_file in os.listdir(schedules_dir):
        # Split filename into parts
        parts = each_file.split("_")

        # Check if this is a valid schedule file for the given device and app
        if (
            len(parts) == 4
            and parts[0] == device
            and parts[1] == APPLICATION_NAME_MAP[application_name]
            and parts[2] == "schedule"
            and parts[3].endswith(".json")
        ):
            num_schedules += 1

    return num_schedules


def select_schedules(device: str, app: str, args_schedules: str | None) -> set[int]:
    """Interactively select schedules or parse from command line args.

    Args:
        device: Device ID
        app: Application name
        args_schedules: Schedule range string from command line args

    Returns:
        Set of selected schedule IDs
    """
    # Get the actual number of schedules for this device-app pair
    max_schedule = get_num_schedules(device, app)
    if max_schedule == 0:
        print(f"Error: No schedules found for {app} on device {device}")
        sys.exit(1)

    if args_schedules:
        schedule_ids = parse_schedule_range(args_schedules)
    else:
        print("\nSelect schedules to run:")
        print("1. All schedules")
        print("2. Range of schedules")
        choice = input("Enter your choice (default: 1): ").strip()

        if choice == "2":
            while True:
                range_str = input(
                    f"Enter schedule range (e.g. '1-5' or '1,3,5', max {max_schedule}): "
                )
                try:
                    schedule_ids = parse_schedule_range(range_str)
                    break
                except argparse.ArgumentTypeError as e:
                    print(f"Error: {e}")
        else:
            schedule_ids = set(range(1, max_schedule + 1))

    # Validate schedule IDs
    invalid_ids = [sid for sid in schedule_ids if not 1 <= sid <= max_schedule]
    if invalid_ids:
        print(
            f"Error: Invalid schedule IDs {invalid_ids}. "
            f"Must be between 1 and {max_schedule}"
        )
        sys.exit(1)

    return schedule_ids
