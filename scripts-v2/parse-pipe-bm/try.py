from load_benchmark import get_benchmark_time, compare_benchmarks
import pandas as pd
import os
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json

HARDWARE_PATH = "data/hardware_config.json"
APPLICATION_PATH = "data/app_config.json"


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


def annotate_schedules_with_timing(
    schedules: List[Schedule], device_id: str, application: str, df: pd.DataFrame
) -> List[Schedule]:
    """
    Annotate schedules with timing information from benchmark data.

    Args:
        schedules: List of Schedule objects
        device_id: Device ID from hardware config
        application: Application name
        df: DataFrame containing benchmark data

    Returns:
        List of schedules with timing information added
    """
    print(f"Using benchmark data for device {device_id}...")

    # First, get baseline times for CPU and GPU
    cpu_baseline_time = None
    gpu_baseline_time = None

    # Get CPU baseline (OMP, Stage 0)
    cpu_baseline_time = get_benchmark_time(df, vendor="OMP", stage=0)
    if cpu_baseline_time is not None:
        print(f"Found CPU baseline time: {cpu_baseline_time:.2f}")
    else:
        print("Warning: CPU baseline data not found")

    # Try to get GPU baseline (CUDA or VK, Stage 0)
    for gpu_backend in ["CUDA", "VK"]:
        gpu_baseline_time = get_benchmark_time(df, vendor=gpu_backend, stage=0)
        if gpu_baseline_time is not None:
            print(f"Found GPU ({gpu_backend}) baseline time: {gpu_baseline_time:.2f}")
            break

    if gpu_baseline_time is None:
        print("Warning: GPU baseline data not found")

    annotated_schedules = []

    # Create a mapping from core types to numeric values
    core_type_to_num = {"little": "0", "medium": "1", "big": "2"}

    for schedule in schedules:
        modified_schedule = False
        chunk_times = [0.0] * len(schedule.chunks)

        for i, (chunk, pu_type, threads) in enumerate(
            zip(schedule.chunks, schedule.pu_types, schedule.pu_threads)
        ):
            start_stage, end_stage = chunk
            chunk_time = 0.0

            # Map pu_type to backend and core_type
            vendor = None
            core = None

            if pu_type.startswith("gpu"):
                # Extract backend from gpu_{backend}
                backend_name = pu_type.split("_")[1]
                if backend_name == "vulkan":
                    vendor = "VK"
                elif backend_name == "cuda":
                    vendor = "CUDA"
            else:
                # CPU core type
                vendor = "OMP"
                core = pu_type
                # Convert core type name to numeric value if it's in our mapping
                if core in core_type_to_num:
                    core = core_type_to_num[core]

            # Sum up time for each stage in the chunk
            for stage in range(start_stage, end_stage + 1):
                # Get benchmark time for current stage
                stage_time = None

                if vendor == "OMP" and threads is not None:
                    # For OMP, we might have data with thread count
                    # The actual column to filter by would depend on your DataFrame structure
                    # This assumes you have a ThreadCount column or similar
                    # You may need to adjust this based on your actual data structure
                    stage_time = get_benchmark_time(
                        df, vendor=vendor, stage=stage, core=core
                    )
                else:
                    # For GPU or if thread count isn't specified
                    stage_time = get_benchmark_time(df, vendor=vendor, stage=stage)

                if stage_time is not None:
                    chunk_time += stage_time
                    modified_schedule = True

                if stage_time is not None:
                    # print(f"Found time for Vendor={vendor}, Stage={stage}, Core={core}: {stage_time:.2f} ms")
                    chunk_time += stage_time
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


if __name__ == "__main__":
    # Load the dataframe
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_pickle(
        os.path.join(base_dir, "parsed_data/3A021JEHN02756_benchmark.pkl")
    )

    # Load configurations
    with open(HARDWARE_PATH, "r") as f:
        hardware_config = json.load(f)

    with open(APPLICATION_PATH, "r") as f:
        application_config = json.load(f)

    device_id = "3A021JEHN02756"
    application = "CifarSparse"

    if device_id not in hardware_config:
        raise ValueError(f"Device ID '{device_id}' not found in hardware config")

    if application not in application_config:
        raise ValueError(f"Application '{application}' not found in application config")

    device_config = hardware_config[device_id]
    num_stages = application_config[application]["num_stages"]

    # Get available processing unit types
    pu_types = get_available_pu_types(device_config)
    max_chunks = len(pu_types)

    print(f"Available PU types: {pu_types}")
    print(f"Maximum chunks: {max_chunks}")

    # Try getting a specific timing from the benchmark data
    vk_stage1_time = get_benchmark_time(df, vendor="VK", stage=1)
    if vk_stage1_time is not None:
        print(f"Example benchmark: VK Stage 1 time: {vk_stage1_time} ms")

    # Generate schedules and annotate with timing information
    try:
        # Generate all possible schedules
        schedules = generate_schedules(device_id, application)
        print(
            f"Generated {len(schedules)} possible schedules for {application} on {device_id}"
        )

        # Annotate schedules with timing information from the dataframe
        annotated_schedules = annotate_schedules_with_timing(
            schedules, device_id, application, df
        )

        if annotated_schedules:
            # Sort schedules by max_chunk_time (ascending)
            sorted_schedules = sorted(
                annotated_schedules, key=lambda s: s.max_chunk_time
            )
            # Limit to top 10 schedules
            top_n = min(10, len(sorted_schedules))
            top_schedules = sorted_schedules[:top_n]

            print(f"\nTop {top_n} schedules by performance (lowest max chunk time):\n")
            for i, schedule in enumerate(top_schedules):
                print_schedule(schedule, i + 1)
                print(f"  Chunk times: {[f'{t:.2f}' for t in schedule.chunk_times]}")
                print(f"  Max chunk time: {schedule.max_chunk_time:.2f}")

                # Print speedup information if available
                if schedule.cpu_speedup:
                    print(f"  CPU speedup: {schedule.cpu_speedup:.2f}x")
                if schedule.gpu_speedup:
                    print(f"  GPU speedup: {schedule.gpu_speedup:.2f}x")

                # Print load balancing metrics if available
                if schedule.load_balance_ratio:
                    print(f"  Load balance ratio: {schedule.load_balance_ratio:.2f}")
                if schedule.load_imbalance_pct:
                    print(f"  Load imbalance: {schedule.load_imbalance_pct:.2f}%")
                if schedule.time_variance:
                    print(f"  Time variance: {schedule.time_variance:.4f}")
                print()
        else:
            print(
                "No schedules could be annotated with timing information. Check your benchmark data."
            )
    except ValueError as e:
        print(f"Error: {e}")
