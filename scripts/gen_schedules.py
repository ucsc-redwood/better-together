import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import itertools
import os
import sys
from pathlib import Path

HARDWARE_PATH = "data/hardware_config.json"
APPLICATION_PATH = "data/app_config.json"

# Add import for query function
from common import init_db, query, ParsedRunName


@dataclass
class Schedule:
    chunks: List[Tuple[int, int]]
    pu_types: List[str]
    pu_threads: List[int]
    chunk_times: List[float]
    max_chunk_time: float
    # Add new fields for baseline data and speedup
    cpu_baseline_time: Optional[float] = None
    gpu_baseline_time: Optional[float] = None
    cpu_speedup: Optional[float] = None
    gpu_speedup: Optional[float] = None
    # Add load balancing indicators
    load_balance_ratio: Optional[float] = None
    load_imbalance_pct: Optional[float] = None
    time_variance: Optional[float] = None


def get_all_possible_chunks(
    num_stages: int, max_chunks: int
) -> List[List[Tuple[int, int]]]:
    """Generate all possible ways to divide stages into continuous chunks"""

    def generate_partitions(n, max_parts):
        # Generate all ways to partition n elements into at most max_parts parts
        if n <= 0 or max_parts <= 0:
            return []
        if max_parts == 1:
            return [[n]]

        result = []
        for i in range(1, n + 1):
            for p in generate_partitions(n - i, max_parts - 1):
                result.append([i] + p)

        # Also include the case where we use fewer than max_parts
        if max_parts > 1:
            result.extend(generate_partitions(n, max_parts - 1))

        return result

    # Get all partitions of the stages
    partitions = generate_partitions(num_stages, max_chunks)

    # Convert partitions to chunks with start-end indices
    all_chunks = []
    for partition in partitions:
        chunks = []
        start_idx = 1  # Stages are 1-indexed

        for part_size in partition:
            end_idx = start_idx + part_size - 1
            chunks.append((start_idx, end_idx))
            start_idx = end_idx + 1

        # Only keep valid partitions (those that use all stages)
        if start_idx > num_stages:
            all_chunks.append(chunks)

    return all_chunks


def get_available_pu_types(device_config: Dict) -> List[str]:
    """Get list of available processing unit types for the device"""
    pu_types = []
    pinnable_cores = device_config["pinnable_cores"]

    for pu_type, count in pinnable_cores.items():
        if count > 0 and pu_type != "gpu":
            pu_types.append(pu_type)

    # Add GPU if available
    if pinnable_cores.get("gpu", 0) > 0:
        # Add the GPU with its backend type
        pu_types.append(f"gpu_{device_config['gpu_backend']}")

    return pu_types


def generate_schedules(device_id: str, application: str) -> List[Schedule]:
    """Generate all possible schedules for the given device and application"""
    # Load configurations
    with open(HARDWARE_PATH, "r") as f:
        hardware_config = json.load(f)

    with open(APPLICATION_PATH, "r") as f:
        application_config = json.load(f)

    if device_id not in hardware_config:
        raise ValueError(f"Device ID '{device_id}' not found in hardware config")

    if application not in application_config:
        raise ValueError(f"Application '{application}' not found in application config")

    device_config = hardware_config[device_id]
    num_stages = application_config[application]["num_stages"]

    # Get available processing unit types
    pu_types = get_available_pu_types(device_config)
    max_chunks = len(pu_types)

    # Generate all possible chunk divisions
    all_chunks = get_all_possible_chunks(num_stages, max_chunks)

    # Generate all possible schedules
    schedules = []

    for chunks in all_chunks:
        num_chunks = len(chunks)

        # Get all possible combinations of PU types
        for pu_type_combo in itertools.combinations(pu_types, num_chunks):
            # Get all permutations of PU types
            for pu_type_perm in itertools.permutations(pu_type_combo):
                # Use all available threads for each PU type
                thread_combo = [
                    (
                        device_config["pinnable_cores"][pu_type.split("_")[0]]
                        if not pu_type.startswith("gpu")
                        else 1
                    )
                    for pu_type in pu_type_perm
                ]

                # Create a schedule
                # For this example, set placeholder values for chunk_times and max_chunk_time
                schedule = Schedule(
                    chunks=chunks,
                    pu_types=list(pu_type_perm),
                    pu_threads=thread_combo,
                    chunk_times=[1.0] * num_chunks,  # Placeholder
                    max_chunk_time=1.0,  # Placeholder
                )
                schedules.append(schedule)

    return schedules


def print_schedule(schedule: Schedule, idx: int = None) -> None:
    """Print a schedule in a readable format"""
    header = f"Schedule {idx}: " if idx is not None else "Schedule: "
    print(header)

    for i, (chunk, pu_type, threads) in enumerate(
        zip(schedule.chunks, schedule.pu_types, schedule.pu_threads)
    ):
        start, end = chunk
        stages = ", ".join(str(s) for s in range(start, end + 1))
        print(
            f"  Chunk {i+1}: Stages [{stages}] → {pu_type} ({threads} thread{'s' if threads > 1 else ''})"
        )

    print()


def annotate_schedules_with_timing(
    schedules: List[Schedule], device_id: str, application: str, benchmark_dir: str
) -> List[Schedule]:
    """
    Annotate schedules with timing information from benchmark data.

    Args:
        schedules: List of Schedule objects
        device_id: Device ID from hardware config
        application: Application name
        benchmark_dir: Directory containing benchmark files

    Returns:
        List of schedules with timing information added
    """
    # Initialize benchmark database
    print(f"Loading benchmark data from {benchmark_dir} for device {device_id}...")
    db = init_db(benchmark_dir, device_id)

    if not db:
        print(f"No benchmark data found for device {device_id}")
        return schedules

    # First, get baseline times for CPU and GPU
    cpu_baseline_time = None
    gpu_baseline_time = None

    # Get CPU baseline (OMP, Stage 0)
    cpu_results = query(db, application=application, backend="OMP", stage=0)
    if cpu_results:
        cpu_baseline_time = cpu_results[0][1]
        print(f"Found CPU baseline time: {cpu_baseline_time:.2f}")
    else:
        print("Warning: CPU baseline data not found")

    # Try to get GPU baseline (CUDA or VK, Stage 0)
    for gpu_backend in ["CUDA", "VK"]:
        gpu_results = query(db, application=application, backend=gpu_backend, stage=0)
        if gpu_results:
            gpu_baseline_time = gpu_results[0][1]
            print(f"Found GPU ({gpu_backend}) baseline time: {gpu_baseline_time:.2f}")
            break

    if gpu_baseline_time is None:
        print("Warning: GPU baseline data not found")

    annotated_schedules = []

    for schedule in schedules:
        modified_schedule = False
        chunk_times = [0.0] * len(schedule.chunks)

        for i, (chunk, pu_type, threads) in enumerate(
            zip(schedule.chunks, schedule.pu_types, schedule.pu_threads)
        ):
            start_stage, end_stage = chunk
            chunk_time = 0.0

            # Map pu_type to backend and core_type
            backend = None
            core_type = None

            if pu_type.startswith("gpu"):
                # Extract backend from gpu_{backend}
                backend_name = pu_type.split("_")[1]
                if backend_name == "vulkan":
                    backend = "VK"
                elif backend_name == "cuda":
                    backend = "CUDA"
            else:
                # CPU core type
                backend = "OMP"
                core_type = pu_type

            # Sum up time for each stage in the chunk
            for stage in range(start_stage, end_stage + 1):
                results = query(
                    db,
                    application=application,
                    backend=backend,
                    stage=stage,
                    core_type=core_type,
                    num_threads=threads,
                )

                if results:
                    # Use the first matching result's time
                    chunk_time += results[0][1]
                    modified_schedule = True
                else:
                    # If we don't have benchmark data for this specific configuration,
                    # we'll try to find data without thread count specified
                    results = query(
                        db,
                        application=application,
                        backend=backend,
                        stage=stage,
                        core_type=core_type,
                    )

                    if results:
                        # Use the first matching result's time
                        chunk_time += results[0][1]
                        modified_schedule = True

            chunk_times[i] = chunk_time

        # Only add schedules where we found timing data
        if modified_schedule:
            max_chunk_time = max(chunk_times) if chunk_times else 0.0

            # Calculate load balancing metrics
            load_balance_ratio = None
            load_imbalance_pct = None
            time_variance = None

            if chunk_times and max_chunk_time > 0:
                min_chunk_time = min(chunk_times)
                avg_chunk_time = sum(chunk_times) / len(chunk_times)

                # Load balance ratio (min/max): closer to 1.0 means better balancing
                if min_chunk_time > 0:
                    load_balance_ratio = min_chunk_time / max_chunk_time

                # Load imbalance percentage: 0% means perfectly balanced
                load_imbalance_pct = (
                    ((max_chunk_time / avg_chunk_time) - 1.0) * 100
                    if avg_chunk_time > 0
                    else None
                )

                # Time variance: lower means better balancing
                if len(chunk_times) > 1:
                    time_variance = sum(
                        (t - avg_chunk_time) ** 2 for t in chunk_times
                    ) / len(chunk_times)

            # Calculate speedups
            cpu_speedup = None
            gpu_speedup = None

            if cpu_baseline_time and max_chunk_time > 0:
                cpu_speedup = cpu_baseline_time / max_chunk_time

            if gpu_baseline_time and max_chunk_time > 0:
                gpu_speedup = gpu_baseline_time / max_chunk_time

            annotated_schedule = Schedule(
                chunks=schedule.chunks,
                pu_types=schedule.pu_types,
                pu_threads=schedule.pu_threads,
                chunk_times=chunk_times,
                max_chunk_time=max_chunk_time,
                cpu_baseline_time=cpu_baseline_time,
                gpu_baseline_time=gpu_baseline_time,
                cpu_speedup=cpu_speedup,
                gpu_speedup=gpu_speedup,
                load_balance_ratio=load_balance_ratio,
                load_imbalance_pct=load_imbalance_pct,
                time_variance=time_variance,
            )
            annotated_schedules.append(annotated_schedule)

    return annotated_schedules


def schedule_to_json(
    schedule: Schedule,
    schedule_id: str,
    device_id: str,
    application: str,
) -> dict:
    """Convert a Schedule object to a JSON-serializable dictionary."""
    chunks_json = []
    for i, (start, end) in enumerate(schedule.chunks):
        stages = list(range(start, end + 1))
        chunk_json = {
            "name": f"chunk{i+1}",
            "hardware": schedule.pu_types[i],
            "threads": schedule.pu_threads[i],
            "stages": stages,
            "time": schedule.chunk_times[i],
        }
        chunks_json.append(chunk_json)

    schedule_json = {
        "schedule": {
            "schedule_id": schedule_id,
            "device_id": device_id,
            "application": application,
            "chunks": chunks_json,
        },
        "max_chunk_time": schedule.max_chunk_time,
    }

    # Add baseline and speedup information if available
    if schedule.cpu_baseline_time is not None:
        schedule_json["cpu_baseline_time"] = schedule.cpu_baseline_time

    if schedule.gpu_baseline_time is not None:
        schedule_json["gpu_baseline_time"] = schedule.gpu_baseline_time

    if schedule.cpu_speedup is not None:
        schedule_json["cpu_speedup"] = schedule.cpu_speedup

    if schedule.gpu_speedup is not None:
        schedule_json["gpu_speedup"] = schedule.gpu_speedup

    # Add load balancing metrics if available
    if schedule.load_balance_ratio is not None:
        schedule_json["load_balance_ratio"] = schedule.load_balance_ratio

    if schedule.load_imbalance_pct is not None:
        schedule_json["load_imbalance_pct"] = schedule.load_imbalance_pct

    if schedule.time_variance is not None:
        schedule_json["time_variance"] = schedule.time_variance

    return schedule_json


def write_schedules_to_json(
    schedules: List[Schedule],
    device_id: str,
    application: str,
    output_dir: str,
) -> None:
    """Write individual schedule JSON files to device/application subdirectories."""
    # Create device/application subdirectories
    output_path = Path(output_dir) / device_id / application
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, schedule in enumerate(schedules, 1):
        schedule_id = f"{device_id}_{application}_schedule_{idx:03d}"
        schedule_json = schedule_to_json(schedule, schedule_id, device_id, application)
        filename = f"schedule_{idx:03d}.json"  # Simplified filename since it's in app/device dir
        file_path = output_path / filename
        with open(file_path, "w") as f:
            json.dump(schedule_json, f, indent=2)

    print(f"Wrote {len(schedules)} schedule files to {output_path}")


def write_all_schedules_to_file(
    schedules: List[Schedule],
    device_id: str,
    application: str,
    output_file: str,
) -> None:
    """Write all schedules to a single JSON file."""
    # Ensure the directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schedules_json = []
    for idx, schedule in enumerate(schedules, 1):
        schedule_id = f"{device_id}_{application}_schedule_{idx:03d}"
        schedule_json = schedule_to_json(schedule, schedule_id, device_id, application)
        schedules_json.append(schedule_json)

    # Create the final JSON structure
    output_json = {
        "device_id": device_id,
        "application": application,
        "total_schedules": len(schedules),
        "schedules": schedules_json,
    }

    with open(output_file, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"Wrote all {len(schedules)} schedules to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all possible schedules for a device and application"
    )
    parser.add_argument(
        "-d",
        "--device_id",
        type=str,
        required=True,
        help="Device ID from hardware_config.json",
    )
    parser.add_argument(
        "-a",
        "--application",
        type=str,
        required=True,
        help="Application name from app_config.json",
    )
    parser.add_argument(
        "-b",
        "--benchmark_dir",
        type=str,
        help="Directory containing benchmark files",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top schedules to display (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Directory to write individual schedule JSON files",
    )
    parser.add_argument(
        "-f",
        "--output_file",
        type=str,
        help="File to write all schedules in a single JSON file",
    )

    args = parser.parse_args()

    try:
        schedules = generate_schedules(args.device_id, args.application)
        print(
            f"Generated {len(schedules)} possible schedules for {args.application} on {args.device_id}"
        )

        sorted_schedules = schedules

        # Annotate schedules with timing information if benchmark directory is provided
        if args.benchmark_dir:
            if not os.path.exists(args.benchmark_dir):
                print(
                    f"Error: Benchmark directory '{args.benchmark_dir}' does not exist"
                )
                sys.exit(1)

            annotated_schedules = annotate_schedules_with_timing(
                schedules, args.device_id, args.application, args.benchmark_dir
            )

            if annotated_schedules:
                # Sort schedules by max_chunk_time (ascending)
                sorted_schedules = sorted(
                    annotated_schedules, key=lambda s: s.max_chunk_time
                )
                # Limit to top N schedules
                sorted_schedules = sorted_schedules[: args.top]

                print(
                    f"\nTop {len(sorted_schedules)} schedules by performance (lowest max chunk time):\n"
                )

                for i, schedule in enumerate(sorted_schedules):
                    print_schedule(schedule, i + 1)
                    print(
                        f"  Chunk times: {[f'{t:.2f}' for t in schedule.chunk_times]}"
                    )
                    print(f"  Max chunk time: {schedule.max_chunk_time:.2f}")

                    # Print speedup information if available
                    if schedule.cpu_speedup:
                        print(f"  CPU speedup: {schedule.cpu_speedup:.2f}x")
                    if schedule.gpu_speedup:
                        print(f"  GPU speedup: {schedule.gpu_speedup:.2f}x")

                    # Print load balancing metrics if available
                    if schedule.load_balance_ratio:
                        print(
                            f"  Load balance ratio: {schedule.load_balance_ratio:.2f}"
                        )
                    if schedule.load_imbalance_pct:
                        print(f"  Load imbalance: {schedule.load_imbalance_pct:.2f}%")
                    if schedule.time_variance:
                        print(f"  Time variance: {schedule.time_variance:.4f}")

                    print()
            else:
                print(
                    "No schedules could be annotated with timing information. Check your benchmark data."
                )
        else:
            # Limit to top N schedules when no benchmark data
            sorted_schedules = schedules[: args.top]
            print(
                f"\nTop {len(sorted_schedules)} possible schedules (without timing information):\n"
            )
            for i, schedule in enumerate(sorted_schedules):
                print_schedule(schedule, i + 1)

        # Write only top N schedules to output directory if specified
        if args.output_dir:
            write_schedules_to_json(
                sorted_schedules, args.device_id, args.application, args.output_dir
            )

        # Write only top N schedules to a single file if specified
        if args.output_file:
            write_all_schedules_to_file(
                sorted_schedules, args.device_id, args.application, args.output_file
            )

    except ValueError as e:
        print(f"Error: {e}")
