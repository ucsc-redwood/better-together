#!/usr/bin/env python3
import pandas as pd
import argparse
import sys
import os


def get_benchmark_time(df, vendor, stage, core=None):
    """
    Get the benchmark time for a specific configuration.

    Args:
        df: DataFrame containing benchmark data
        vendor: The vendor (e.g., 'VK', 'OMP')
        stage: The stage number (as string or int)
        core: The core number (optional, used for OMP vendor)

    Returns:
        The MeanTime_ms value if found, None otherwise
    """
    # Convert stage to string if it's not already
    stage = str(stage)

    # Create a query for the specific configuration
    mask = (df["Vendor"] == vendor) & (df["Stage"] == stage)

    # Add core filter if provided and vendor is OMP
    if core is not None and vendor == "OMP":
        core = str(core)
        mask = mask & (df["Core"] == core)
    elif vendor == "VK" and core is not None:
        print(f"Warning: Core parameter ignored for vendor {vendor}")

    # Get the matching rows
    result = df[mask]

    if len(result) == 0:
        print(
            f"No data found for Vendor={vendor}, Stage={stage}{f', Core={core}' if core else ''}"
        )
        return None
    elif len(result) > 1:
        print(f"Multiple results found ({len(result)} rows). Returning the first one.")

    # Return the timing
    return result["MeanTime_ms"].iloc[0] if not result.empty else None


def compare_benchmarks(df, configs):
    """
    Compare benchmark times for multiple configurations.

    Args:
        df: DataFrame containing benchmark data
        configs: List of tuples (vendor, stage, core) or (vendor, stage)

    Returns:
        DataFrame with comparison results
    """
    results = []

    for config in configs:
        if len(config) == 2:
            vendor, stage = config
            core = None
        elif len(config) == 3:
            vendor, stage, core = config
        else:
            print(f"Invalid config format: {config}")
            continue

        time = get_benchmark_time(df, vendor, stage, core)

        result = {"Vendor": vendor, "Stage": stage, "Core": core, "MeanTime_ms": time}
        results.append(result)

    return pd.DataFrame(results)


def load_and_display_pickle(pickle_path):
    """
    Load a pickle file containing a pandas DataFrame and display it.

    Args:
        pickle_path: Path to the pickle file
    """
    try:
        # Check if file exists
        if not os.path.exists(pickle_path):
            print(f"Error: File '{pickle_path}' not found.")
            return False

        # Load the DataFrame from pickle
        df = pd.read_pickle(pickle_path)

        # Print information about the DataFrame
        print("\n=== DataFrame Summary ===")
        print(f"Rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print("\n=== DataFrame Content ===")

        # Display the DataFrame with better formatting
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.expand_frame_repr", False)
        print(df)

        # Print basic statistics
        print("\n=== Numeric Statistics ===")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("No numeric columns found for statistics.")

        return True

    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Load and display a pickled pandas DataFrame."
    )
    parser.add_argument("pickle_file", help="Path to the pickle file to load")
    parser.add_argument(
        "--filter",
        help="Filter the DataFrame by a column value (format: column=value)",
        default=None,
    )
    # Add parameters for the get_benchmark_time function
    parser.add_argument(
        "--vendor", help="Filter by vendor (e.g., VK, OMP)", default=None
    )
    parser.add_argument("--stage", help="Filter by stage number", default=None)
    parser.add_argument("--core", help="Filter by core number (for OMP)", default=None)
    parser.add_argument(
        "--compare",
        help="Compare multiple configurations specified as 'vendor1:stage1[:core1],vendor2:stage2[:core2],...'",
        default=None,
    )

    args = parser.parse_args()

    # Load the DataFrame
    try:
        df = pd.read_pickle(args.pickle_file)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # Apply filter if provided
    if args.filter:
        try:
            column, value = args.filter.split("=")
            # Try to convert value to appropriate type
            try:
                # Try as numeric
                value = float(value) if "." in value else int(value)
            except ValueError:
                # Keep as string
                pass

            df = df[df[column] == value]
            print(f"Filtered by {column}={value}, showing {len(df)} rows")
        except Exception as e:
            print(f"Error applying filter: {e}")

    # If vendor and stage are provided, get specific benchmark time
    if args.vendor and args.stage:
        time = get_benchmark_time(df, args.vendor, args.stage, args.core)
        if time is not None:
            config_str = f"Vendor={args.vendor}, Stage={args.stage}"
            if args.core:
                config_str += f", Core={args.core}"
            print(f"Benchmark time for {config_str}: {time} ms")

    # If compare is provided, compare multiple configurations
    if args.compare:
        try:
            configs = []
            for config_str in args.compare.split(","):
                parts = config_str.split(":")
                if len(parts) == 2:
                    vendor, stage = parts
                    configs.append((vendor, stage))
                elif len(parts) == 3:
                    vendor, stage, core = parts
                    configs.append((vendor, stage, core))
                else:
                    print(f"Invalid config format: {config_str}")
                    continue

            print("\n=== Benchmark Comparison ===")
            comparison = compare_benchmarks(df, configs)
            # Add speedup column relative to the first row
            if len(comparison) > 1 and not comparison["MeanTime_ms"].isna().any():
                baseline = comparison["MeanTime_ms"].iloc[0]
                comparison["Speedup"] = baseline / comparison["MeanTime_ms"]
            print(comparison)
        except Exception as e:
            print(f"Error comparing benchmarks: {e}")

    # Display the DataFrame if no specific query was made
    if not (args.vendor and args.stage) and not args.compare:
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.expand_frame_repr", False)
        print(df)


if __name__ == "__main__":
    main()
