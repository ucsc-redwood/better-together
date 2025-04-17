import os
import shutil
import subprocess
import argparse
import sys


class ScheduleRunner:
    def __init__(self):
        self.log_dir = "logs"
        self.tmp_dir = "tmp_folder"
        self.accumulated_file = "accumulated_time.txt"

    def run_schedule(self, device, app):
        """Run benchmarking schedules for the given device and app."""
        if not device or not app:
            print("Error: Both device and app must be specified")
            return False

        # Setup files and directories
        log_file = f"{device}_{app}_schedules.log"
        self._cleanup_files(log_file)

        # Run the benchmark command
        schedule_url = f"http://192.168.1.204:8080/{device}_{app}_vk_schedules.json"
        cmd = self._build_benchmark_command(device, app, schedule_url)

        try:
            with open(log_file, "w") as log_file_handle:
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                log_file_handle.write(result.stdout)
                print(result.stdout)
            return True
        except subprocess.SubprocessError as e:
            print(f"Error running schedule: {e}")
            return False

    def run_schedule_part_2(self, device, app):
        """Process logs and accumulate benchmark results."""
        if not device or not app:
            print("Error: Both device and app must be specified")
            return False

        log_file = f"{device}_{app}_schedules.log"
        tmp2_file = "tmp2.txt"

        if not os.path.exists(log_file):
            print(f"Error: Log file {log_file} does not exist")
            return False

        # Run the processing script
        cmd = [
            "python3",
            "scripts/plot/schedule_exe.py",
            "--output-dir",
            f"{self.tmp_dir}/",
            log_file,
        ]

        try:
            with open(tmp2_file, "w") as out:
                subprocess.run(cmd, stdout=out, text=True, check=True)

            # Extract and accumulate execution time information
            self._extract_execution_time(tmp2_file)

            # Display accumulated results
            self._display_accumulated_results()
            return True
        except subprocess.SubprocessError as e:
            print(f"Error processing schedule results: {e}")
            return False

    def _cleanup_files(self, log_file):
        """Clean up log files and temporary directories."""
        if os.path.exists(log_file):
            os.remove(log_file)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _build_benchmark_command(self, device, app, schedule_url):
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
            "10",
        ]

    def _extract_execution_time(self, tmp2_file):
        """Extract execution time from temporary file and add to accumulated results."""
        with open(tmp2_file, "r") as f, open(self.accumulated_file, "a") as acc:
            for line in f:
                if "Total execution time:" in line:
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
    parser.add_argument("task", choices=["serve", "run", "part2"], help="Task to run")
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

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    runner = ScheduleRunner()

    if args.task == "run":
        if not args.device or not args.app:
            print("Error: Both --device and --app are required for 'run'")
            sys.exit(1)
        success = runner.run_schedule(args.device, args.app)
        if not success:
            sys.exit(1)
    elif args.task == "part2":
        if not args.device or not args.app:
            print("Error: Both --device and --app are required for 'part2'")
            sys.exit(1)
        success = runner.run_schedule_part_2(args.device, args.app)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
