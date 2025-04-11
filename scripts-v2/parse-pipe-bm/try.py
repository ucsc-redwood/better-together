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


if __name__ == "__main__":
    # Load your dataframe
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

    print(pu_types)
    print(max_chunks)

    # Generate all possible chunk divisions
    all_chunks = get_all_possible_chunks(num_stages, max_chunks)

    for chunks in all_chunks:
        print(chunks)

    try:
        schedules = generate_schedules(device_id, application)
        print(
            f"Generated {len(schedules)} possible schedules for {application} on {device_id}"
        )
    except ValueError as e:
        print(f"Error: {e}")

    # # Get a specific timing
    # vk_stage1_time = get_benchmark_time(df, vendor="VK", stage=1)

    # print(f"VK Stage 1 time: {vk_stage1_time} ms")
