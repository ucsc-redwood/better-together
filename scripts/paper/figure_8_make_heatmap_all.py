#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    Group the DataFrame by 'stage' and compute mean and std for the specified columns.
    Returns three DataFrames: mean, std, cv (coefficient of variation).
    """
    group = df.groupby("stage")
    mean_df = group[value_cols].mean()
    std_df = group[value_cols].std()
    cv_df = std_df.divide(mean_df).fillna(0)
    return mean_df, std_df, cv_df


def plot_bar_chart(
    mean_normal, std_normal, mean_fully, std_fully, device, out_folder, app, backend_col
):
    available_cores = mean_normal.columns.tolist()
    stages = mean_normal.index.astype(int).tolist()
    x = np.arange(len(stages))
    width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    titles = ["Normal Benchmark", "Fully Benchmark"]
    data_pairs = [(mean_normal, std_normal), (mean_fully, std_fully)]

    for ax, (mean_df, std_df), title in zip(axs, data_pairs, titles):
        for i, pu in enumerate(available_cores):
            pos = x - (1.5 * width) + i * width
            ax.bar(pos, mean_df[pu], width, yerr=std_df[pu], capsize=5, label=pu)
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_xlabel("Stage")
        ax.set_title(title)
        ax.legend()

    axs[0].set_ylabel("Performance (ms per task)")
    plt.suptitle(f"Benchmark Performance for {app} ({backend_col}) on {device}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(out_folder, f"{device}_{app}_{backend_col}_bar_chart.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved bar chart: {outpath}")


def plot_heatmap(diff_df, device, out_folder, app, backend_col):
    available_cores = diff_df.columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(diff_df.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(available_cores)))
    ax.set_xticklabels(available_cores)
    ax.set_yticks(np.arange(len(diff_df.index)))
    ax.set_yticklabels(diff_df.index.astype(str))
    # ax.set_xlabel("Processing Unit")
    ax.set_ylabel("Stage")
    ax.set_title(f"Fully - Normal Diff for {app} ({backend_col}) on {device}")

    for i in range(diff_df.shape[0]):
        for j in range(diff_df.shape[1]):
            val = diff_df.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    outpath = os.path.join(out_folder, f"{device}_{app}_{backend_col}_heatmap.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved heatmap: {outpath}")


def plot_ratio_heatmap(ratio_df, device, out_folder, app, backend_col):
    available_cores = ratio_df.columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(ratio_df.values, cmap="coolwarm", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(available_cores)))
    ax.set_xticklabels(available_cores)
    ax.set_yticks(np.arange(len(ratio_df.index)))
    ax.set_yticklabels(ratio_df.index.astype(str))
    # ax.set_xlabel("Processing Unit")
    ax.set_ylabel("Stage")
    ax.set_title(f"Fully/Normal Ratio for {app} ({backend_col}) on {device}")

    for i in range(ratio_df.shape[0]):
        for j in range(ratio_df.shape[1]):
            val = ratio_df.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    outpath = os.path.join(out_folder, f"{device}_{app}_{backend_col}_ratio.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved ratio heatmap: {outpath}")


def process_data_pair(folder_root, device, app, backend, exclude_stages=None):
    """Process a single device-app-backend pair and return the ratio DataFrame."""
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
        
        mean_n, _, _ = aggregate_data(df_normal, available_cores)
        mean_f, _, _ = aggregate_data(df_fully, available_cores)
        
        if exclude_stages:
            idx = mean_n.index.astype(int)
            mask = ~idx.isin(exclude_stages)
            mean_n, mean_f = mean_n[mask], mean_f[mask]
        
        # Calculate ratio
        ratio = mean_f.divide(mean_n)
        
        # Generate individual ratio heatmap for records
        plot_ratio_heatmap(ratio, device, device_folder, app, backend_col)
        
        return ratio, f"{device}\n{app}\n{backend_col}"
    
    except Exception as e:
        print(f"Error processing {device}/{app}/{backend}: {str(e)}")
        return None, None


def plot_combined_ratio_heatmaps(ratios, titles, out_folder):
    """Create a 2x3 grid of ratio heatmaps."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    for i, (ratio_df, title) in enumerate(zip(ratios, titles)):
        if ratio_df is None:
            axs[i].text(
                0.5, 0.5, f"No data for\n{title}", ha="center", va="center", fontsize=12
            )
            axs[i].axis("off")
            continue
            
        available_cores = ratio_df.columns.tolist()
        im = axs[i].imshow(
            ratio_df.values, cmap="coolwarm", aspect="auto", vmin=0, vmax=2
        )
        
        axs[i].set_xticks(np.arange(len(available_cores)))
        axs[i].set_xticklabels(available_cores, rotation=0, ha="right")
        axs[i].set_yticks(np.arange(len(ratio_df.index)))
        axs[i].set_yticklabels(ratio_df.index.astype(str))
        
        axs[i].set_title(title)
        
        # Add text annotations
        for y in range(ratio_df.shape[0]):
            for x in range(ratio_df.shape[1]):
                val = ratio_df.iat[y, x]
                axs[i].text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=7)
    
    # Add a colorbar to the figure
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Fully/Normal Ratio")
    
    plt.suptitle("Fully/Normal Execution Time Ratio Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    outpath = os.path.join(out_folder, "combined_ratio_heatmaps.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved combined ratio heatmaps: {outpath}")


def aggregate_device_data(device_ratios, device_name):
    """Aggregate ratio data for a single device across all applications."""
    if not device_ratios or all(r is None for r in device_ratios):
        return None
    
    valid_ratios = [r for r in device_ratios if r is not None]
    if not valid_ratios:
        return None
    
    # Check if all DataFrames have the same shape and indices
    ref_ratio = valid_ratios[0]
    ref_shape = ref_ratio.shape
    ref_index = ref_ratio.index
    ref_columns = ref_ratio.columns
    
    aligned_ratios = []
    for ratio in valid_ratios:
        if ratio.shape != ref_shape or not ratio.index.equals(ref_index) or not ratio.columns.equals(ref_columns):
            # Reindex to match reference ratio
            try:
                ratio = ratio.reindex(index=ref_index, columns=ref_columns, fill_value=np.nan)
            except Exception as e:
                print(f"Warning: Could not align ratios for {device_name}: {str(e)}")
                continue
        aligned_ratios.append(ratio)
    
    if not aligned_ratios:
        return None
    
    # Average the ratios
    try:
        avg_ratio = pd.concat(aligned_ratios).groupby(level=0).mean()
        return avg_ratio
    except Exception as e:
        print(f"Error aggregating data for {device_name}: {str(e)}")
        return None


def plot_device_aggregated_heatmaps(device_ratios, device_names, out_folder):
    """Create a 1x4 grid of aggregated heatmaps by device."""
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    min_val, max_val = 0, 2  # Default min/max values
    
    # Find common min/max for better comparison
    valid_ratios = [r for r in device_ratios if r is not None]
    if valid_ratios:
        all_values = np.concatenate([r.values.flatten() for r in valid_ratios])
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) > 0:
            min_val = max(0, np.percentile(all_values, 5))
            max_val = min(2, np.percentile(all_values, 95))
    
    for i, (ratio_df, device_name) in enumerate(zip(device_ratios, device_names)):
        if ratio_df is None:
            axs[i].text(
                0.5, 0.5, f"No data for\n{device_name}", 
                ha="center", va="center", fontsize=18
            )
            axs[i].axis("off")
            continue
            
        available_cores = ratio_df.columns.tolist()
        im = axs[i].imshow(
            ratio_df.values, cmap="coolwarm", aspect="auto", vmin=min_val, vmax=max_val
        )
        
        axs[i].set_xticks(np.arange(len(available_cores)))
        axs[i].set_xticklabels(available_cores, rotation=45, ha="right", fontsize=16)
        axs[i].set_yticks(np.arange(len(ratio_df.index)))
        axs[i].set_yticklabels(ratio_df.index.astype(str), fontsize=16)
        
        # Add axis labels with large fonts
        # axs[i].set_xlabel("Processing Unit", fontsize=18)
        if i == 0:  # Only add y-label to the first subplot to save space
            axs[i].set_ylabel("Stage", fontsize=18)
        
        axs[i].set_title(f"{device_name}", fontsize=20)
        
        # Add text annotations
        for y in range(ratio_df.shape[0]):
            for x in range(ratio_df.shape[1]):
                val = ratio_df.iat[y, x]
                if not np.isnan(val):
                    axs[i].text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=14)
    
    # Add a colorbar to the figure
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Fully/Normal Ratio", fontsize=18)
    cbar.ax.tick_params(labelsize=16)  # Larger tick labels on colorbar
    
    # plt.suptitle("Fully/Normal Execution Time Ratio by Device", fontsize=22)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    
    outpath = os.path.join(out_folder, "device_aggregated_heatmaps.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved device aggregated heatmaps: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark CSVs and generate figures."
    )
    parser.add_argument("--log_folder", required=True, help="Root folder with CSVs")
    parser.add_argument(
        "--exclude_stages", default="", help="Comma-separated stages to exclude"
    )
    parser.add_argument("--app", help="Application name (required with --individual)")
    parser.add_argument(
        "--backend",
        choices=BACKEND_COL_MAP.keys(),
        help="Backend (required with --individual)",
    )
    parser.add_argument(
        "--device", help="Device identifier (required with --individual)"
    )

    args = parser.parse_args()
    
    folder_root = args.log_folder
    
    # Parse excluded stages
    exclude = []
    if args.exclude_stages:
        try:
            exclude = [int(x) for x in args.exclude_stages.split(",") if x.strip()]
            print(f"Excluding stages: {exclude}")
        except ValueError:
            print(f"Invalid --exclude_stages: {args.exclude_stages}")
            sys.exit(1)
    
    # Define device-application-backend combinations
    devices_mapping = {
        "3A021JEHN02756": ["cifar-sparse", "cifar-dense", "tree"],  # Pixel
        "9b034f1b": ["cifar-sparse", "cifar-dense", "tree"],        # S22
        "jetson": ["cifar-sparse", "cifar-dense", "tree"],          # Jetson
        "jetsonlowpower": ["cifar-sparse", "cifar-dense", "tree"]       # Jetson Low Power
    }
    
    # Backend to use for each device
    device_backends = {
        "3A021JEHN02756": "vk",  # Pixel uses Vulkan
        "9b034f1b": "vk",        # S22 uses Vulkan
        "jetson": "cu",          # Jetson uses CUDA
        "jetsonlowpower": "cu"       # Jetson Low Power uses CUDA
    }
    
    # Store ratios by device
    device_all_ratios = {device: [] for device in devices_mapping.keys()}
    
    # Process all device-app-backend combinations
    for device, apps in devices_mapping.items():
        backend = device_backends[device]
        for app in apps:
            ratio_df, _ = process_data_pair(folder_root, device, app, backend, exclude)
            if ratio_df is not None:
                device_all_ratios[device].append(ratio_df)
    
    # Aggregate ratios by device
    aggregated_device_ratios = []
    device_display_names = []
    
    for device, ratios in device_all_ratios.items():
        display_name = DEVICE_NAME_MAP.get(device, device)
        device_display_names.append(display_name)
        agg_ratio = aggregate_device_data(ratios, display_name)
        aggregated_device_ratios.append(agg_ratio)
    
    # Create aggregated device heatmap
    plot_device_aggregated_heatmaps(aggregated_device_ratios, device_display_names, folder_root)


if __name__ == "__main__":
    main()
