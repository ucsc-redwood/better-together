#!/usr/bin/env python3
from pathlib import Path
import argparse
from helpers import (
    ALL_DEVICES,
    ALL_BENCHMARKS,
    interactive_select,
    RAW_BENCHMARK_PATH,
    run_command,
)


def get_json_filename(benchmark: str, device: str) -> str:
    """Generate benchmark JSON filename based on benchmark name and device."""
    parts = benchmark.split("-")
    if len(parts) == 4:
        return (
            f"BM_{parts[1].capitalize()}{parts[2].capitalize()}_"
            f"{parts[3].upper()}_{device}.json"
        )
    return f"BM_{parts[1].capitalize()}_{parts[2].upper()}_{device}.json"


def run_benchmark(benchmark: str, device: str, output_dir: Path) -> None:
    """Run a single benchmark on a device and collect results."""
    print(f"\n--- Running {benchmark} on device: {device} ---")

    # Push executable to device
    exe_path = f"./build/android/arm64-v8a/release/{benchmark}"
    device_path = f"/data/local/tmp/{benchmark}"
    run_command(f"adb -s {device} push {exe_path} {device_path}", hide_output=True)

    # Run benchmark
    run_command(f"adb -s {device} shell {device_path} " f"--device {device}")

    # Pull results
    json_file = get_json_filename(benchmark, device)
    run_command(
        f"adb -s {device} pull /data/local/tmp/{json_file} {output_dir}/",
        hide_output=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on Android devices")
    parser.add_argument(
        "--benchmark",
        "-b",
        help="Benchmark to run",
        choices=ALL_BENCHMARKS,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device to run on",
        choices=ALL_DEVICES,
    )
    args = parser.parse_args()

    # Select benchmarks and devices
    benchmark = (
        args.benchmark
        if args.benchmark
        else interactive_select(ALL_BENCHMARKS, "benchmark")
    )
    device = args.device if args.device else interactive_select(ALL_DEVICES, "device")

    print(f"\nRunning benchmark: {benchmark} on device: {device}")

    # Setup output directory
    output_dir = Path(RAW_BENCHMARK_PATH)
    output_dir.mkdir(exist_ok=True)

    # Run benchmark
    run_command(f"xmake b {benchmark}")

    run_benchmark(benchmark, device, output_dir)


if __name__ == "__main__":
    main()
