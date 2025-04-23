import os
import shutil
import subprocess
import argparse
import sys
import re
import time
import datetime
import pandas as pd


class ScheduleRunner:
    def __init__(self):
        # Base directories
        self.base_dir = "new_data"
        self.date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self.data_dir = os.path.join(self.base_dir, self.date_str)
        self.schedules_dir = os.path.join(self.data_dir, "schedules")
        self.results_dir = os.path.join(self.data_dir, "results")
        self.tmp_dir = "tmp_folder"

        # Ensure base directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.schedules_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # HTTP server parameters
        self.server_port = 8080
        self.server_process = None

    def get_app_dir(self, app):
        """Get directory for a specific application."""
        app_dir = os.path.join(self.data_dir, app)
        os.makedirs(app_dir, exist_ok=True)
        return app_dir

    def get_device_results_dir(self, device, app):
        """Get results directory for a specific device and application."""
        device_dir = os.path.join(self.results_dir, f"{device}_{app}")
        os.makedirs(device_dir, exist_ok=True)
        return device_dir

    def run_benchmark(self, app, repeat=3, device=None):
        """Run benchmarks and collect data for the given app.
        The device parameter is optional since bm.py detects all connected devices."""
        app_dir = self.get_app_dir(app)
        device_str = f" on {device}" if device else " on all connected devices"
        print(f"\n=== Running benchmark for {app}{device_str} ===")

        # Build the benchmark command
        cmd = [
            "python3",
            "scripts/collect/bm.py",
            "--log_folder",
            app_dir,
            "--repeat",
            str(repeat),
            "--target",
            f"bm-fully-{app}-vk",
        ]

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Benchmark data collected for {app} in {app_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark: {e}")
            return False

    def create_heatmaps(self, app, exclude_stages=""):
        """Generate heatmaps from benchmark data."""
        app_dir = self.get_app_dir(app)
        print(f"\n=== Creating heatmaps for {app} ===")

        cmd = ["python3", "scripts/plot/normal_vs_fully_heat.py", "--folder", app_dir]

        if exclude_stages:
            cmd.extend(["--exclude_stages", exclude_stages])

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Heatmaps created for {app} in {app_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating heatmaps: {e}")
            return False

    def generate_schedules(self, device, app, num_schedules=30, backend=None):
        """Generate schedules using the Z3 optimizer.
        
        Args:
            device: Device ID
            app: Application name
            num_schedules: Number of schedules to generate
            backend: Backend to use ('vk' for Vulkan, 'cu' for CUDA, None for auto-detect)
        """
        app_dir = self.get_app_dir(app)
        
        # If backend not specified, auto-detect from CSV
        if backend is None:
            csv_path = os.path.join(app_dir, f"{device}_fully.csv")
            if not os.path.exists(csv_path):
                print(f"Error: CSV file {csv_path} not found. Run benchmarks first.")
                return False
                
            # Read CSV to detect if CUDA values are present
            try:
                df = pd.read_csv(csv_path)
                # Check if CUDA column has non-zero values
                if 'cuda' in df.columns and df['cuda'].sum() > 0:
                    backend = 'cu'
                    print(f"Auto-detected CUDA data in CSV. Using CUDA backend.")
                else:
                    backend = 'vk'
                    print(f"Using Vulkan backend (no CUDA data detected).")
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                print(f"Defaulting to Vulkan backend.")
                backend = 'vk'
        
        schedule_file = os.path.join(
            self.schedules_dir, f"{device}_{app}_{backend}_schedules.json"
        )
        print(f"\n=== Generating schedules for {app} on {device} using {backend} backend ===")

        csv_path = os.path.join(app_dir, f"{device}_fully.csv")
        if not os.path.exists(csv_path):
            print(f"Error: CSV file {csv_path} not found. Run benchmarks first.")
            return False

        cmd = [
            "python3",
            "scripts/gen/schedule.py",
            "--csv_folder",
            app_dir,
            "--device",
            device,
            "--app",
            app,
            "--backend",
            backend,
            "-n",
            str(num_schedules),
            "--output_folder",
            self.schedules_dir,
        ]

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Schedules generated and saved to {schedule_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error generating schedules: {e}")
            return False

    def start_server(self):
        """Start HTTP server to serve schedule files."""
        if self.server_process and self.server_process.poll() is None:
            print(f"Server already running on port {self.server_port}")
            return True

        print(f"\n=== Starting HTTP server on port {self.server_port} ===")
        try:
            self.server_process = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "http.server",
                    "--bind",
                    "0.0.0.0",
                    "--directory",
                    self.schedules_dir,
                    str(self.server_port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(1)  # Give server time to start
            if self.server_process.poll() is None:
                print(f"Server started successfully on port {self.server_port}")
                return True
            else:
                print(
                    f"Failed to start server: {self.server_process.stderr.read().decode()}"
                )
                return False
        except Exception as e:
            print(f"Error starting server: {e}")
            return False

    def stop_server(self):
        """Stop the HTTP server."""
        if self.server_process and self.server_process.poll() is None:
            print("\n=== Stopping HTTP server ===")
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            self.server_process = None
            print("Server stopped")
            return True
        return False

    def run_schedule(self, device, app, n_to_run=30, backend=None):
        """Run benchmarking schedules for the given device and app."""
        device_results_dir = self.get_device_results_dir(device, app)

        if not device or not app:
            print("Error: Both device and app must be specified")
            return False
        
        # If backend not specified, auto-detect
        if backend is None:
            schedule_vk = os.path.join(
                self.schedules_dir, f"{device}_{app}_vk_schedules.json"
            )
            schedule_cu = os.path.join(
                self.schedules_dir, f"{device}_{app}_cu_schedules.json"
            )
            
            if os.path.exists(schedule_cu):
                backend = 'cu'
                print(f"Auto-detected CUDA schedule file. Using CUDA backend.")
            elif os.path.exists(schedule_vk):
                backend = 'vk'
                print(f"Using Vulkan backend (no CUDA schedule detected).")
            else:
                print(f"No schedule files found for device {device} and app {app}.")
                print(f"Defaulting to Vulkan backend.")
                backend = 'vk'

        # Ensure server is running
        if not self.start_server():
            print("Error: Failed to start HTTP server. Cannot run schedules.")
            return False

        # Generate log file name with incremental suffix if needed
        base_log_file = os.path.join(
            device_results_dir, f"{device}_{app}_{backend}_schedules.log"
        )
        log_file = self._get_incremental_filename(base_log_file)

        # Create tmp directory if it doesn't exist
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Run the benchmark command
        schedule_url = (
            f"http://192.168.1.204:{self.server_port}/{device}_{app}_{backend}_schedules.json"
        )
        cmd = self._build_benchmark_command(device, app, schedule_url, n_to_run, backend)

        try:
            print(f"Running command: {' '.join(cmd)}")
            print(f"Logging to: {log_file}")

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

            print(f"Schedule log saved to {log_file}")
            return True
        except subprocess.SubprocessError as e:
            print(f"Error running schedule: {e}")
            return False

    def parse_results(self, device, app):
        """Parse and analyze schedule execution results."""
        device_results_dir = self.get_device_results_dir(device, app)
        schedule_file = os.path.join(
            self.schedules_dir, f"{device}_{app}_vk_schedules.json"
        )

        if not os.path.exists(schedule_file):
            print(f"Error: Schedule file {schedule_file} not found.")
            return False

        print(f"\n=== Parsing results for {app} on {device} ===")

        cmd = [
            "python3",
            "scripts/parse_schedules_by_widest.py",
            device_results_dir,
            "--model",
            schedule_file,
            "--output",
            os.path.join(device_results_dir, "analysis"),
        ]

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(
                f"Results parsed and saved to {os.path.join(device_results_dir, 'analysis')}"
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error parsing results: {e}")
            return False

    def _get_incremental_filename(self, base_filename):
        """Generate an incremental filename if the base name already exists."""
        # Extract name and extension
        name_parts = base_filename.rsplit(".", 1)
        base_name = name_parts[0]
        extension = f".{name_parts[1]}" if len(name_parts) > 1 else ""

        # Always start with _000 suffix
        counter = 0
        while True:
            new_filename = f"{base_name}_{counter:03d}{extension}"
            if not os.path.exists(new_filename):
                return new_filename
            counter += 1

    def _build_benchmark_command(self, device, app, schedule_url, n_to_run, backend='vk'):
        """Build the benchmark command with appropriate parameters."""
        return [
            "xmake",
            "r",
            f"bm-gen-logs-{app}-{backend}",
            "-l",
            "off",
            "--device-to-measure",
            device,
            "--schedule-url",
            schedule_url,
            "--n-schedules-to-run",
            str(n_to_run),
        ]

    def execute_pipeline(
        self, device, app, steps=None, repeat=3, num_schedules=30, exclude_stages="", backend=None
    ):
        """Execute the entire pipeline or specific steps for a device/app pair."""
        if steps is None:
            steps = ["benchmark", "heatmap", "schedule", "run", "parse"]

        success = True

        if "benchmark" in steps:
            success = success and self.run_benchmark(app, repeat, device)

        if "heatmap" in steps and success:
            success = success and self.create_heatmaps(app, exclude_stages)

        if "schedule" in steps and success:
            success = success and self.generate_schedules(device, app, num_schedules, backend)

        if "run" in steps and success:
            success = success and self.run_schedule(device, app, num_schedules, backend)

        if "parse" in steps and success:
            success = success and self.parse_results(device, app)

        if "server" in steps:
            if steps == ["server"]:
                # Just start the server if that's the only step
                success = success and self.start_server()
            else:
                # Stop the server if it was part of other steps
                self.stop_server()

        return success


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarking and scheduling tasks for heterogeneous systems."
    )
    parser.add_argument(
        "task",
        choices=[
            "benchmark",
            "heatmap",
            "schedule",
            "run",
            "parse",
            "server",
            "pipeline",
        ],
        help="Task to run",
    )

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
        "--repeat", type=int, default=3, help="Number of times to repeat benchmarks"
    )

    parser.add_argument(
        "-n",
        "--num-schedules",
        type=int,
        default=30,
        help="Number of schedules to generate or run",
    )

    parser.add_argument(
        "--exclude-stages",
        default="",
        help="Comma-separated list of stages to exclude from heatmaps",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["benchmark", "heatmap", "schedule", "run", "parse", "server"],
        help="Steps to execute in the pipeline (default: all)",
    )

    parser.add_argument(
        "--backend",
        choices=["vk", "cu", "auto"],
        default="auto",
        help="Backend to use (vk=Vulkan, cu=CUDA, auto=auto-detect)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    runner = ScheduleRunner()

    # For most tasks, device and app are required
    tasks_requiring_device_app = ["schedule", "run", "parse", "pipeline"]
    tasks_requiring_app_only = ["benchmark", "heatmap"]

    if args.task in tasks_requiring_device_app and (not args.device or not args.app):
        print(f"Error: Both --device and --app are required for '{args.task}'")
        sys.exit(1)
    elif args.task in tasks_requiring_app_only and not args.app:
        print(f"Error: --app is required for '{args.task}'")
        sys.exit(1)

    # Convert 'auto' backend to None for auto-detection
    backend = None if args.backend == 'auto' else args.backend

    # Execute the requested task
    success = False

    if args.task == "benchmark":
        success = runner.run_benchmark(args.app, args.repeat, args.device)

    elif args.task == "heatmap":
        success = runner.create_heatmaps(args.app, args.exclude_stages)

    elif args.task == "schedule":
        success = runner.generate_schedules(args.device, args.app, args.num_schedules, backend)

    elif args.task == "run":
        success = runner.run_schedule(args.device, args.app, args.num_schedules, backend)

    elif args.task == "parse":
        success = runner.parse_results(args.device, args.app)

    elif args.task == "server":
        success = runner.start_server()

        print("Server is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            runner.stop_server()

    elif args.task == "pipeline":
        success = runner.execute_pipeline(
            args.device,
            args.app,
            args.steps,
            args.repeat,
            args.num_schedules,
            args.exclude_stages,
            backend,
        )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
