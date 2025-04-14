#!/usr/bin/env python3
import sys
import re
import math
import statistics

# ANSI color escape codes.
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"


def colored_text(text, color):
    return f"{color}{text}{COLOR_RESET}"


def parse_line(line):
    """
    Parse a line of the form:
      PROCESSOR=Little|COUNT=100|TOTAL=1987.96|AVG=19.8796|...
    Return a dictionary with keys (as strings) and numbers where possible.
    """
    fields = line.strip().split("|")
    data = {}
    for field in fields:
        if "=" in field:
            key, value = field.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


def group_lines(lines, group_size):
    """Return chunks (groups) of lines of length group_size."""
    return [lines[i : i + group_size] for i in range(0, len(lines), group_size)]


def pearson_correlation(x, y):
    """
    Compute Pearson's correlation coefficient between lists x and y.
    Returns None if not computable.
    """
    n = len(x)
    if n < 2:
        return None
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    try:
        sdx = statistics.stdev(x)
        sdy = statistics.stdev(y)
    except statistics.StatisticsError:
        return None
    if sdx == 0 or sdy == 0:
        return None
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
    return cov / (sdx * sdy)


def main():
    input_file = "full_output.txt"
    try:
        with open(input_file, "r") as f:
            # Remove empty lines.
            all_lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Cannot open {input_file}")
        sys.exit(1)

    # We assume each set consists of 4 lines.
    if len(all_lines) % 4 != 0:
        print("ERROR: The number of lines is not a multiple of 4.")
        sys.exit(1)

    # Each set: 4 lines parsed into a list of dictionaries.
    sets = [[parse_line(line) for line in group] for group in group_lines(all_lines, 4)]

    # Every two sets form one comparison group.
    if len(sets) % 2 != 0:
        print("ERROR: The number of sets is not even (each group should have 2 sets).")
        sys.exit(1)
    groups = group_lines(sets, 2)

    print(f"Found {len(groups)} groups in the input (expected 9 groups).\n")

    # For correlation analysis, we will collect for each processor the paired AVG values
    # from set A (Non-Full) and set B (Full) across all groups.
    corr_data = {"Big": [], "GPU": [], "Little": [], "Medium": []}

    # Process each group.
    for group_index, group in enumerate(groups, start=1):
        setA, setB = group  # First set and second set in the group.
        avg_A = {}
        avg_B = {}
        # Build mapping for each set using the 'PROCESSOR' field.
        for item in setA:
            proc = item.get("PROCESSOR")
            if proc and "AVG" in item:
                avg_A[proc] = item["AVG"]
        for item in setB:
            proc = item.get("PROCESSOR")
            if proc and "AVG" in item:
                avg_B[proc] = item["AVG"]

        # We'll list all processors that appear.
        all_procs = sorted(set(avg_A.keys()) | set(avg_B.keys()))

        # Print header for the group.
        print(f"Group {group_index} Comparison (Set A = Non-Full; Set B = Full):")
        print("-" * 80)
        header = f"{'Processor':10s} {'Set A AVG':>10s} {'Set B AVG':>10s} {'Diff':>12s} {'% Diff':>10s} {'Ratio':>10s}"
        print(header)
        print("-" * 80)
        # For each processor, calculate difference and percentage difference.
        for proc in all_procs:
            a_val = avg_A.get(proc)
            b_val = avg_B.get(proc)
            # If either value is missing, skip.
            if a_val is None or b_val is None:
                diff = "N/A"
                pct_diff = "N/A"
                ratio = "N/A"
            else:
                diff_val = b_val - a_val
                ratio_val = b_val / a_val if a_val != 0 else float("inf")
                pct_diff_val = (diff_val / a_val) * 100 if a_val != 0 else 0

                # Colorize the percentage difference: if negative (i.e. full is lower => improvement), use green;
                # if positive (worse), use red; zero is left uncolored.
                pct_diff_str = f"{pct_diff_val:+.1f}%"
                if pct_diff_val < 0:
                    pct_diff_str = colored_text(pct_diff_str, COLOR_GREEN)
                elif pct_diff_val > 0:
                    pct_diff_str = colored_text(pct_diff_str, COLOR_RED)

                diff = f"{diff_val:.6f}"
                pct_diff = pct_diff_str
                ratio = f"{ratio_val:.6f}"

                # Also, for correlation analysis, store the pair for this processor.
                corr_data.setdefault(proc, []).append((a_val, b_val))
            # Print the table row.
            print(
                f"{proc:10s} {a_val if a_val is not None else 'N/A':10} {b_val if b_val is not None else 'N/A':10} {diff:12} {pct_diff:10} {ratio:10}"
            )
        print("-" * 80)
        print()  # Blank line between groups

    # Now, perform correlation analysis for each processor.
    print("\nCorrelation Analysis (across all groups):")
    print("-" * 60)
    print(f"{'Processor':10s} {'Pearson r':>12s} {'n (# pairs)':>12s}")
    print("-" * 60)
    for proc, pairs in corr_data.items():
        # We expect pairs to be a list of (avgA, avgB) values.
        if pairs:
            x_vals, y_vals = zip(*pairs)
            # Compute Pearson r manually.
            r = pearson_correlation(x_vals, y_vals)
            n = len(pairs)
            # Format r with 3 decimal places.
            r_str = f"{r:.3f}" if r is not None else "N/A"
            print(f"{proc:10s} {r_str:12} {n:12d}")
        else:
            print(f"{proc:10s} {'N/A':12} {0:12d}")
    print("-" * 60)


if __name__ == "__main__":
    main()
