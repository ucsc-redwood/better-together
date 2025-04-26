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
    ax.set_xlabel("Processing Unit")
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
    ax.set_xlabel("Processing Unit")
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
        axs[i].set_xticklabels(available_cores, rotation=45, ha="right")
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

    # Hardcoded 2x3 grid of device-app-backend pairs
    pairs = [
        ("3A021JEHN02756", "cifar-sparse", "vk"),
        ("3A021JEHN02756", "cifar-dense", "vk"),
        ("3A021JEHN02756", "tree", "vk"),
        ("9b034f1b", "cifar-sparse", "vk"),
        ("9b034f1b", "cifar-dense", "vk"),
        ("9b034f1b", "tree", "vk"),
    ]

    ratios = []
    titles = []

    for device, app, backend in pairs:
        ratio_df, title = process_data_pair(folder_root, device, app, backend, exclude)
        ratios.append(ratio_df)
        titles.append(title)

    # Create combined figure
    plot_combined_ratio_heatmaps(ratios, titles, folder_root)


if __name__ == "__main__":
    main()
