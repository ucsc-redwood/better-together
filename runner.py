import os
import shutil
import subprocess
import argparse
import sys
import re


class ScheduleRunner:
    def __init__(self):
        self.log_dir = "logs"
        self.tmp_dir = "tmp_folder"
        self.accumulated_file = "accumulated_time.txt"

    def run_schedule(self, device, app, n_to_run):
        """Run benchmarking schedules for the given device and app."""
        if not device or not app:
            print("Error: Both device and app must be specified")
            return False

        # Setup files and directories
        log_file = f"{device}_{app}_schedules.log"
        self._cleanup_files(log_file)

        # Run the benchmark command
        schedule_url = f"http://192.168.1.204:8080/{device}_{app}_vk_schedules.json"
        cmd = self._build_benchmark_command(device, app, schedule_url, n_to_run)

        try:
            print(f"Running command: {' '.join(cmd)}")

            # Use Popen to get real-time output
            with open(log_file, "w") as log_file_handle:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Process and display output in real-time
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")  # Print to console in real-time
                    log_file_handle.write(line)  # Write to log file

                # Wait for process to complete
                return_code = process.wait()
                if return_code != 0:
                    print(f"Command failed with return code {return_code}")
                    return False

            return True
        except subprocess.SubprocessError as e:
            print(f"Error running schedule: {e}")
            return False

    def _cleanup_files(self, log_file):
        """Clean up log files and temporary directories."""
        if os.path.exists(log_file):
            os.remove(log_file)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _build_benchmark_command(self, device, app, schedule_url, n_to_run):
        """Build the benchmark command with appropriate parameters."""
        return [
            "xmake",
            "r",
            f"bm-gen-logs-{app}-vk",
            "-l",
            "off",
            "--device-to-measure",
            device,
            "--schedule-url",
            schedule_url,
            "--n-schedules-to-run",
            str(n_to_run),
        ]

    def _extract_execution_time_with_uid(self, tmp2_file, log_file):
        """Extract execution time and schedule UID from log files and add to accumulated results."""
        # First extract UIDs from the main log file
        uids = {}
        with open(log_file, "r") as f:
            schedule_id = None
            for line in f:
                # Match schedule ID line
                id_match = re.search(r"Running schedule (\d+)", line)
                if id_match:
                    schedule_id = id_match.group(1)

                # Match UID line and associate with the current schedule_id
                uid_match = re.search(r"Schedule_UID: ([^\s]+)", line)
                if uid_match and schedule_id is not None:
                    uids[schedule_id] = uid_match.group(1)

        # Extract execution times and add UIDs
        with open(tmp2_file, "r") as f, open(self.accumulated_file, "a") as acc:
            schedule_id = None
            for line in f:
                # Try to extract schedule ID if present
                id_match = re.search(r"Schedule (\d+)", line)
                if id_match:
                    schedule_id = id_match.group(1)

                # Extract execution time and add UID if available
                if "Total execution time:" in line:
                    if schedule_id and schedule_id in uids:
                        uid = uids[schedule_id]
                        # Append UID to the execution time line
                        line = line.rstrip() + f" [UID: {uid}]\n"
                    acc.write(line)

    def _display_accumulated_results(self):
        """Display the accumulated benchmark results."""
        try:
            with open(self.accumulated_file, "r") as acc:
                print(acc.read())
        except FileNotFoundError:
            print(f"No accumulated results found in {self.accumulated_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run heterogeneous schedule tasks.")
    parser.add_argument("task", choices=["run", "part2"], help="Task to run")
    parser.add_argument(
        "--device",
        choices=["3A021JEHN02756", "9b034f1b", "jetson", "jetsonlowpower"],
        help="Target device",
    )
    parser.add_argument(
        "--app",
        choices=["cifar-dense", "cifar-sparse", "tree"],
        help="Application to run",
    )

    parser.add_argument(
        "--n-schedules-to-run",
        type=int,
        default=10,
        help="Number of schedules to run",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    runner = ScheduleRunner()

    if args.task == "run":
        if not args.device or not args.app:
            print("Error: Both --device and --app are required for 'run'")
            sys.exit(1)
        success = runner.run_schedule(args.device, args.app, args.n_schedules_to_run)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
