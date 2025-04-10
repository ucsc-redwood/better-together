import re
import argparse
from collections import defaultdict


def parse_benchmark_file(file_path):
    # Dictionary to store results: stage_id -> num_threads -> core_type -> time
    results = defaultdict(lambda: defaultdict(dict))

    with open(file_path, "r") as f:
        for line in f:
            # Skip header lines, empty lines, baseline benchmarks and SKIPPED lines
            if (
                not line.strip()
                or "SKIPPED" in line
                or "BM_run_OMP_baseline" in line
                or "-" in line[:5]
            ):
                continue

            # Parse valid benchmark lines
            match = re.match(
                r"BM_run_OMP_stage/(\d+)/(\d+)/(\d+)\s+(\d+\.?\d*)\s+ms", line
            )
            if match:
                stage_id, num_threads, core_type, time = match.groups()
                stage_id = int(stage_id)
                num_threads = int(num_threads)
                core_type = int(core_type)
                time = float(time)

                results[stage_id][num_threads][core_type] = time

    return results


def print_results(results, threads_filter=None, core_filter=None, stage_filter=None):
    core_types = {0: "little", 1: "mid", 2: "big"}
    core_type_ids = {"little": 0, "mid": 1, "big": 2}

    print("Parsed Benchmark Results:")
    print("=" * 60)
    print(f"{'Stage':<6} {'Threads':<8} {'Core Type':<10} {'Time (ms)':<10}")
    print("-" * 60)

    for stage_id in sorted(results.keys()):
        if stage_filter is not None and stage_id != stage_filter:
            continue

        for num_threads in sorted(results[stage_id].keys()):
            if threads_filter is not None and num_threads != threads_filter:
                continue

            for core_type in sorted(results[stage_id][num_threads].keys()):
                if core_filter is not None:
                    filter_id = core_type_ids.get(core_filter.lower())
                    if filter_id is None or core_type != filter_id:
                        continue

                time = results[stage_id][num_threads][core_type]
                core_name = core_types.get(core_type, f"type-{core_type}")
                print(f"{stage_id:<6} {num_threads:<8} {core_name:<10} {time:<10.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Parse and filter benchmark results")
    parser.add_argument(
        "--file", "-f", default="in.txt", help="Input benchmark file path"
    )
    parser.add_argument("--threads", "-t", type=int, help="Filter by thread count")
    parser.add_argument(
        "--core",
        "-c",
        choices=["little", "mid", "big"],
        help="Filter by core type (little, mid, big)",
    )
    parser.add_argument("--stage", "-s", type=int, help="Filter by stage id")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.file

    results = parse_benchmark_file(file_path)
    print_results(
        results,
        threads_filter=args.threads,
        core_filter=args.core,
        stage_filter=args.stage,
    )
