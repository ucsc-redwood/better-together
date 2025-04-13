#!/usr/bin/env python3
import sys
import os


def parse_line(line):
    """
    Parse a single raw line such as:
      PROCESSOR=Little|COUNT=100|TOTAL=1943.48|AVG=19.4348|...
    Returns a dictionary of key-value pairs with numbers converted.
    """
    record = {}
    # Split the line into key=value tokens
    parts = line.strip().split("|")
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        # Attempt to convert to int or float if possible
        try:
            if "." in value:
                record[key] = float(value)
            else:
                record[key] = int(value)
        except ValueError:
            record[key] = value
    return record


def read_raw_file(filepath):
    """
    Read the raw data file and return a list of lines.
    Ignores empty lines.
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_data_structure(lines):
    """
    Given the list of lines, build the data structure.
    Every eight lines form one stage:
       - The first 4 lines are the "non-fully occupied" set
       - The next 4 lines are the "fully occupied" set
    Each set is stored as a dictionary keyed by the processor type.
    Returns a list of groups (stages).
    """
    if len(lines) % 8 != 0:
        raise ValueError(
            "Number of lines in file is not a multiple of 8. Check the raw data format."
        )

    groups = []
    num_groups = len(lines) // 8
    for i in range(num_groups):
        # Extract lines for the current group/stage
        group_lines = lines[i * 8 : (i + 1) * 8]
        # First four lines: non fully occupied set
        non_fully = {}
        for line in group_lines[:4]:
            record = parse_line(line)
            # Remove the processor type from the record and use it as a key
            processor = record.pop("PROCESSOR", None)
            if processor:
                non_fully[processor] = record
            else:
                raise ValueError("Missing PROCESSOR field in line: " + line)
        # Next four lines: fully occupied set
        fully = {}
        for line in group_lines[4:]:
            record = parse_line(line)
            processor = record.pop("PROCESSOR", None)
            if processor:
                fully[processor] = record
            else:
                raise ValueError("Missing PROCESSOR field in line: " + line)
        # Append the stage to the groups list
        groups.append({"non_fully": non_fully, "fully": fully})
    return groups


def query_processor(groups, stage_index, occupancy_type, processor_type):
    """
    Utility function to query the metric dictionary.
    Parameters:
      - groups: the data structure list returned by build_data_structure.
      - stage_index: 0-indexed stage number.
      - occupancy_type: either "non_fully" or "fully".
      - processor_type: string such as "Little", "Medium", "Big", "GPU".
    Returns the dictionary of metrics for that processor.
    """
    try:
        stage = groups[stage_index]
    except IndexError:
        raise IndexError(
            f"Stage index {stage_index} is out of bounds (only {len(groups)} stages available)"
        )
    if occupancy_type not in stage:
        raise ValueError(f"Occupancy type must be one of: {list(stage.keys())}")
    processor_data = stage[occupancy_type].get(processor_type)
    if processor_data is None:
        raise ValueError(
            f"Processor '{processor_type}' not found in stage {stage_index} ({occupancy_type})"
        )
    return processor_data


def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "3A021JEHN02756-cifar-sparse.raw")

    try:
        lines = read_raw_file(filepath)
    except Exception as e:
        print("Error reading file:", e)
        sys.exit(1)

    try:
        data_groups = build_data_structure(lines)
    except Exception as e:
        print("Error processing data:", e)
        sys.exit(1)

    # print(data_groups)
    for group in data_groups:
        print("=" * 80)
        for processor, metrics in group["non_fully"].items():
            for key, val in metrics.items():
                # print(f"  {key}: {val}")
                # only print the key if it contains "AVG"
                if "AVG" in key:
                    print(f"{processor}: {val}")


if __name__ == "__main__":
    main()
