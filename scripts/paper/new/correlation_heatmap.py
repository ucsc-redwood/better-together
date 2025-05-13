#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import sys

# Map backend flag to CSV column name
BACKEND_COL_MAP = {
    "vk": "vulkan",
    "cu": "cuda",
}

# Map device ID to device name for display
DEVICE_NAME_MAP = {
    "3A021JEHN02756": "Google Pixel",
    "9b034f1b": "OnePlus",
    "jetson": "Jetson",
    "jetsonlowpower": "Jetson (low power)",
}


def get_available_cores(df, backend_col):
    """Get list of available cores from the DataFrame by checking for non-zero values."""
    all_cores = ["little", "medium", "big", backend_col]
    available_cores = []
    for core in all_cores:
        if core in df.columns and not df[core].eq(0).all():
            available_cores.append(core)
    return available_cores


def aggregate_data(df, value_cols):
    """
    Group the DataFrame by 'stage' and compute mean for the specified columns.
    Returns a DataFrame of means.
    """
    group = df.groupby("stage")
    mean_df = group[value_cols].mean()
    return mean_df


def process_data_pair(folder_root, device, app, backend, exclude_stages=None):
    """Process a single device-app-backend pair and return the ratio DataFrame and title."""
    backend_col = BACKEND_COL_MAP[backend]
    device_folder = os.path.join(folder_root, device, app, backend)

    if not os.path.isdir(device_folder):
        print(f"Warning: Folder {device_folder} does not exist.")
        return None, None

    normal_csv = os.path.join(device_folder, "normal.csv")
    fully_csv = os.path.join(device_folder, "fully.csv")

    if not os.path.exists(normal_csv) or not os.path.exists(fully_csv):
        print(f"Warning: CSV files not found in {device_folder}")
        return None, None

    try:
        df_normal = pd.read_csv(normal_csv)
        df_fully = pd.read_csv(fully_csv)

        # Get available cores from the data
        available_cores = get_available_cores(df_normal, backend_col)
        if not available_cores:
            print(f"Warning: No valid cores found in data for {device}/{app}/{backend}")
            return None, None

        mean_n = aggregate_data(df_normal, available_cores)
        mean_f = aggregate_data(df_fully, available_cores)

        if exclude_stages:
            idx = mean_n.index.astype(int)
            mask = ~idx.isin(exclude_stages)
            mean_n = mean_n[mask]
            mean_f = mean_f[mask]

        # Calculate ratio
        ratio = mean_f.divide(mean_n)
        return ratio, f"{device}\n{app}\n{backend_col}"

    except Exception as e:
        print(f"Error processing {device}/{app}/{backend}: {e}")
        return None, None


def aggregate_device_data(device_ratios):
    """Aggregate ratio data for a single device across all applications."""
    valid_ratios = [r for r in device_ratios if r is not None]
    if not valid_ratios:
        return None

    # Reference DataFrame shape, index, and columns
    ref = valid_ratios[0]
    ref_index = ref.index
    ref_columns = ref.columns

    aligned = []
    for ratio in valid_ratios:
        if (
            ratio.shape != ref.shape
            or not ratio.index.equals(ref_index)
            or not ratio.columns.equals(ref_columns)
        ):
            ratio = ratio.reindex(
                index=ref_index, columns=ref_columns, fill_value=np.nan
            )
        aligned.append(ratio)

    # Concatenate and average
    try:
        concatenated = pd.concat(aligned, axis=0)
        avg_ratio = concatenated.groupby(level=0).mean()
        return avg_ratio
    except Exception as e:
        print(f"Error aggregating device data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark CSVs and compute ratios."
    )
    parser.add_argument("--log_folder", required=True, help="Root folder with CSVs")
    parser.add_argument(
        "--exclude_stages", default="", help="Comma-separated stages to exclude"
    )
    args = parser.parse_args()

    folder_root = args.log_folder
    exclude = []
    if args.exclude_stages:
        try:
            exclude = [int(x) for x in args.exclude_stages.split(",") if x.strip()]
        except ValueError:
            print(f"Invalid --exclude_stages: {args.exclude_stages}")
            sys.exit(1)

    devices_mapping = {
        "3A021JEHN02756": ["cifar-sparse", "cifar-dense", "tree"],
        "9b034f1b": ["cifar-sparse", "cifar-dense", "tree"],
        "jetson": ["cifar-sparse", "cifar-dense", "tree"],
        "jetsonlowpower": ["cifar-sparse", "cifar-dense", "tree"],
    }

    device_backends = {
        "3A021JEHN02756": "vk",
        "9b034f1b": "vk",
        "jetson": "cu",
        "jetsonlowpower": "cu",
    }

    device_all_ratios = {d: [] for d in devices_mapping}

    for device, apps in devices_mapping.items():
        backend = device_backends[device]
        for app in apps:
            ratio, title = process_data_pair(folder_root, device, app, backend, exclude)
            if ratio is not None:
                device_all_ratios[device].append(ratio)

    aggregated_device_ratios = {}
    for device, ratios in device_all_ratios.items():
        agg = aggregate_device_data(ratios)
        name = DEVICE_NAME_MAP.get(device, device)
        aggregated_device_ratios[name] = agg

    # Output aggregated ratios
    for device_name, df in aggregated_device_ratios.items():
        if df is not None:
            print(f"Aggregated ratio for {device_name}:")
            print(df)
        else:
            print(f"No data to aggregate for {device_name}")

    # Calculate and print average across all stages for each PU type
    print("\nAverage across all stages for each PU type:")
    for device_name, df in aggregated_device_ratios.items():
        if df is not None:
            avg = df.mean()
            print(f"\n{device_name}:")
            print(avg)
            # print(f"\n{device_name}:")
            # for pu, value in avg.items():
            #     print(f"{pu}: {value:.4f}")


  


if __name__ == "__main__":
    main()
