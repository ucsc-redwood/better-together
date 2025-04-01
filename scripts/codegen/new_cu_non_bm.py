#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path

###############################################################################
# Map your known application names to a relevant C++ namespace if needed
###############################################################################
APP_NAMESPACE_MAP = {
    "CifarDense": "cifar_dense",
    "CifarSparse": "cifar_sparse",
    "Tree": "tree",
}

# Hardcoded list of CUDA-capable devices
CUDA_DEVICES = ["jetson", "jetsonlowpower"]  # Example device ID  # For testing


###############################################################################
# Utility to generate the non-benchmark code for CUDA applications
###############################################################################
def generate_non_benchmark_code_cuda(schedule_json, application):
    """
    Generate the non-benchmark CUDA approach with normal timing:
      - cuda::CudaManager mgr
      - init_appdata<app_namespace::AppData>(mr, num_tasks)
      - chunk<Task, app_namespace::AppData>(..., mgr)
    """
    sched = schedule_json["schedule"]
    schedule_id = sched["schedule_id"]
    chunks = sched["chunks"]

    # E.g., "schedule_jetson_CifarDense_schedule_001"
    func_name = f"schedule_{schedule_id}"

    # Convert app name to a namespace, or fallback
    app_ns = APP_NAMESPACE_MAP.get(application, "cifar_dense")

    lines = []
    lines.append(f"static void {func_name}() {{")
    lines.append("  cuda::CudaManager mgr;")
    lines.append("")
    lines.append("  constexpr size_t num_tasks = 100;")
    lines.append("")
    lines.append("  auto mr = &mgr.get_mr();")
    lines.append("")
    lines.append("  // Preallocate data for all tasks")
    lines.append(
        f"  auto preallocated_data = init_appdata<{app_ns}::AppData>(mr, num_tasks);"
    )
    lines.append("")
    lines.append(
        "  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);"
    )
    lines.append("")
    lines.append("  cudaEvent_t start, stop;")
    lines.append("  float milliseconds = 0;")
    lines.append("  cudaEventCreate(&start);")
    lines.append("  cudaEventCreate(&stop);")
    lines.append("  cudaEventRecord(start, 0);")
    lines.append("")
    lines.append(
        "  // ---------------------------------------------------------------------"
    )
    lines.append("  // Automatically generated from schedule JSON")
    lines.append("")

    lines.append('  nvtxRangePushA("Compute");')

    n = len(chunks)
    # Define the intermediate queues
    for i in range(n - 1):
        lines.append(f"  moodycamel::ConcurrentQueue<Task *> q_{i}_{i+1};")
    lines.append("")

    # Build thread calls
    for i, ch in enumerate(chunks):
        hw = ch["hardware"]
        threads = ch["threads"]
        stages = ch["stages"]
        stage_min = min(stages)
        stage_max = max(stages)

        # input Q / output Q
        inQ = "q_input" if i == 0 else f"q_{i-1}_{i}"
        outQ = "nullptr" if i == n - 1 else f"&q_{i}_{i+1}"

        if hw == "gpu_cuda":
            run_fun = f"cuda::run_multiple_stages<{stage_min}, {stage_max}>"
            line = (
                f"std::thread t{i+1}([&]() {{ "
                f"chunk<Task, {app_ns}::AppData>({inQ}, {outQ}, {run_fun}, mgr); }});"
            )
        else:
            # CPU path
            proc_map = {
                "little": "ProcessorType::kLittleCore",
                "medium": "ProcessorType::kMediumCore",
                "big": "ProcessorType::kBigCore",
            }
            if hw not in proc_map:
                raise ValueError(f"Unknown hardware type: {hw}")

            proc_type = proc_map[hw]
            run_fun = f"omp::run_multiple_stages<{stage_min}, {stage_max}, {proc_type}, {threads}>"
            line = (
                f"std::thread t{i+1}([&]() {{ "
                f"chunk<Task, {app_ns}::AppData>({inQ}, {outQ}, {run_fun}, mgr); }});"
            )

        lines.append(f"  {line}")

    # Join threads
    lines.append("")
    for i in range(n):
        lines.append(f"  t{i+1}.join();")

    lines.append("  nvtxRangePop();")

    lines.append("")
    lines.append(
        "  // ---------------------------------------------------------------------"
    )
    lines.append("")
    lines.append("  cudaEventRecord(stop, 0);")
    lines.append("  cudaEventSynchronize(stop);")
    lines.append("  cudaEventElapsedTime(&milliseconds, start, stop);")
    lines.append("  cudaEventDestroy(start);")
    lines.append("  cudaEventDestroy(stop);")
    lines.append("")
    lines.append("  double avg_task_time = milliseconds / num_tasks;")
    lines.append(
        f'  std::cout << "Average time per task for {schedule_id}: " << avg_task_time << " ms" << std::endl;'
    )
    lines.append("}")
    lines.append("")

    return lines, func_name


###############################################################################
# Utility to generate the special "Tree" code
###############################################################################


def generate_non_benchmark_code_cuda_tree(schedule_json):
    """
    The specialized approach for 'Tree' schedules:
      - tree::SafeAppData instead of AppData
    """
    sched = schedule_json["schedule"]
    schedule_id = sched["schedule_id"]
    chunks = sched["chunks"]

    # E.g., "schedule_9b034f1b_CifarDense_schedule_001"
    func_name = f"schedule_{schedule_id}"

    lines = []
    lines.append(f"static void {func_name}() {{")
    lines.append("  cuda::CudaManager mgr;")
    lines.append("")
    lines.append("  constexpr size_t num_tasks = 100;")
    lines.append("")
    lines.append("  auto mr = &mgr.get_mr();")
    lines.append("")
    lines.append("  // Preallocate data for all tasks")
    lines.append(
        f"  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);"
    )
    lines.append("")
    lines.append(
        "  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);"
    )
    lines.append("")
    lines.append("  cudaEvent_t start, stop;")
    lines.append("  float milliseconds = 0;")
    lines.append("  cudaEventCreate(&start);")
    lines.append("  cudaEventCreate(&stop);")
    lines.append("  cudaEventRecord(start, 0);")
    lines.append("")
    lines.append(
        "  // ---------------------------------------------------------------------"
    )
    lines.append("  // Automatically generated from schedule JSON")
    lines.append("")

    lines.append('  nvtxRangePushA("Compute");')

    n = len(chunks)
    # Define the intermediate queues
    for i in range(n - 1):
        lines.append(f"  moodycamel::ConcurrentQueue<Task *> q_{i}_{i+1};")
    lines.append("")

    # Build thread calls
    for i, ch in enumerate(chunks):
        hw = ch["hardware"]
        threads = ch["threads"]
        stages = ch["stages"]
        stage_min = min(stages)
        stage_max = max(stages)

        # input Q / output Q
        inQ = "q_input" if i == 0 else f"q_{i-1}_{i}"
        outQ = "nullptr" if i == n - 1 else f"&q_{i}_{i+1}"

        if hw == "gpu_cuda":
            run_fun = f"cuda::run_multiple_stages<{stage_min}, {stage_max}>"
            line = (
                f"std::thread t{i+1}([&]() {{ "
                f"chunk<Task, tree::SafeAppData>({inQ}, {outQ}, {run_fun}, mgr); }});"
            )
        else:
            # CPU path
            proc_map = {
                "little": "ProcessorType::kLittleCore",
                "medium": "ProcessorType::kMediumCore",
                "big": "ProcessorType::kBigCore",
            }
            if hw not in proc_map:
                raise ValueError(f"Unknown hardware type: {hw}")

            proc_type = proc_map[hw]
            run_fun = f"omp::run_multiple_stages<{stage_min}, {stage_max}, {proc_type}, {threads}>"
            line = (
                f"std::thread t{i+1}([&]() {{ "
                f"chunk<Task, tree::SafeAppData>({inQ}, {outQ}, {run_fun}, mgr); }});"
            )

        lines.append(f"  {line}")

    # Join threads
    lines.append("")
    for i in range(n):
        lines.append(f"  t{i+1}.join();")

    lines.append("  nvtxRangePop();")

    lines.append("")
    lines.append(
        "  // ---------------------------------------------------------------------"
    )
    lines.append("")
    lines.append("  cudaEventRecord(stop, 0);")
    lines.append("  cudaEventSynchronize(stop);")
    lines.append("  cudaEventElapsedTime(&milliseconds, start, stop);")
    lines.append("  cudaEventDestroy(start);")
    lines.append("  cudaEventDestroy(stop);")
    lines.append("")
    lines.append("  double avg_task_time = milliseconds / num_tasks;")
    lines.append(
        f'  std::cout << "Average time per task for {schedule_id}: " << avg_task_time << " ms" << std::endl;'
    )
    lines.append("}")
    lines.append("")

    return lines, func_name


###############################################################################
# Main code generator
###############################################################################
def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python new_cu_non_bm.py <root_dir> <application> <output_file.cuh>"
        )
        print("Example:")
        print(
            "  python new_cu_non_bm.py data/schedule_files/ CifarDense generated_code_non_bm.cuh"
        )
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    application = sys.argv[2]
    output_path = Path(sys.argv[3])

    lines_out = []
    # Basic includes
    lines_out.append("#pragma once")
    lines_out.append("")
    lines_out.append("#include <numeric>")
    lines_out.append("#include <thread>")
    lines_out.append("")
    lines_out.append("#include <nvToolsExt.h>")
    lines_out.append("#include <cuda_runtime.h>")
    lines_out.append("")
    lines_out.append('#include "../templates.hpp"')
    lines_out.append('#include "../templates_cu.hpp"')
    lines_out.append('#include "run_stages.hpp"')
    lines_out.append('#include "task.hpp"')
    lines_out.append("")
    lines_out.append("")

    # We'll define a struct for schedule_table entries in a common namespace:
    lines_out.append("namespace generated_schedules {")
    lines_out.append("using run_func_t = void (*)();")
    lines_out.append("struct ScheduleRecord {")
    lines_out.append("  const char *name;")
    lines_out.append("  run_func_t func;")
    lines_out.append("};")
    lines_out.append("}  // namespace generated_schedules")
    lines_out.append("")

    # Iterate through hardcoded CUDA devices list
    for device in CUDA_DEVICES:
        device_dir = root_dir / device
        if not device_dir.is_dir():
            continue

        # Check if this device has subdir for the chosen application
        app_dir = device_dir / application
        if not app_dir.is_dir():
            continue

        schedule_files = sorted(app_dir.glob("schedule_*.json"))
        if not schedule_files:
            continue

        # Start device namespace
        device_ns = f"device_{device}"
        lines_out.append(f"namespace {device_ns} {{")

        # We'll store (schedule_id, function_name) for the table
        schedule_records = []

        for json_path in schedule_files:
            with open(json_path, "r") as f:
                schedule_json = json.load(f)

            sched = schedule_json["schedule"]
            schedule_id = sched["schedule_id"]
            chunks = sched["chunks"]

            # Map gpu_vulkan to gpu_cuda for CUDA generation
            for chunk in chunks:
                if chunk["hardware"] == "gpu_vulkan":
                    chunk["hardware"] = "gpu_cuda"

            # Skip unknown hardware if encountered
            valid_hw = {"gpu_cuda", "little", "medium", "big"}
            skip_it = False
            for c in chunks:
                if c["hardware"] not in valid_hw:
                    print(
                        f"Skipping {schedule_id} for device '{device}' due to unknown hardware {c['hardware']}"
                    )
                    skip_it = True
                    break
            if skip_it:
                continue

            # Now generate code with the correct function, depending on 'application'
            if application == "Tree":
                func_lines, func_name = generate_non_benchmark_code_cuda_tree(
                    schedule_json
                )
            else:
                func_lines, func_name = generate_non_benchmark_code_cuda(
                    schedule_json, application
                )

            lines_out.extend(func_lines)
            schedule_records.append((schedule_id, func_name))

        # Build device's schedule_table
        lines_out.append("")
        lines_out.append("// Table of schedules for this device:")
        lines_out.append(
            "static generated_schedules::ScheduleRecord schedule_table[] = {"
        )
        for sid, fname in schedule_records:
            lines_out.append(f'    {{"{sid}", &{fname}}},')
        lines_out.append("};")
        lines_out.append(
            "static const size_t schedule_count = sizeof(schedule_table) / sizeof(schedule_table[0]);"
        )
        lines_out.append(f"}}  // namespace {device_ns}\n")

    # Done. Write the file
    with open(output_path, "w") as outf:
        outf.write("\n".join(lines_out))
        outf.write("\n")

    print(f"Done. Generated non-benchmark CUDA code has been written to {output_path}")


if __name__ == "__main__":
    main()
