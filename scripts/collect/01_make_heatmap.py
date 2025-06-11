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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark CSVs and generate figures."
    )
    parser.add_argument("--log_folder", required=True, help="Root folder with CSVs")
    parser.add_argument(
        "--app", required=True, help="Application name (e.g., cifar-sparse)"
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=BACKEND_COL_MAP.keys(),
        help="Backend: vk or cu",
    )
    parser.add_argument("--device", required=True, help="Device identifier")
    parser.add_argument(
        "--exclude_stages", default="", help="Comma-separated stages to exclude"
    )
    args = parser.parse_args()

    folder_root = args.log_folder
    app = args.app
    backend = args.backend
    device = args.device
    backend_col = BACKEND_COL_MAP[backend]

    # Define path to device/app/backend folder
    device_folder = os.path.join(folder_root, device, app, backend)
    if not os.path.isdir(device_folder):
        print(f"Error: Folder {device_folder} does not exist.")
        sys.exit(0)

    normal_csv = os.path.join(device_folder, "normal.csv")
    fully_csv = os.path.join(device_folder, "fully.csv")
    if not os.path.exists(normal_csv):
        print(f"Error: CSV file {normal_csv} not found.")
        sys.exit(0)
    if not os.path.exists(fully_csv):
        print(f"Error: CSV file {fully_csv} not found.")
        sys.exit(0)

    try:
        df_normal = pd.read_csv(normal_csv)
        df_fully = pd.read_csv(fully_csv)

        # Get available cores from the data
        available_cores = get_available_cores(df_normal, backend_col)
        print(f"Available cores for {device}: {available_cores}")

        exclude = []
        if args.exclude_stages:
            try:
                exclude = [int(x) for x in args.exclude_stages.split(",") if x.strip()]
                print(f"Excluding stages: {exclude}")
            except ValueError:
                print(f"Invalid --exclude_stages: {args.exclude_stages}")
                sys.exit(1)

        mean_n, std_n, cv_n = aggregate_data(df_normal, available_cores)
        mean_f, std_f, cv_f = aggregate_data(df_fully, available_cores)

        if exclude:
            idx = mean_n.index.astype(int)
            mask = ~idx.isin(exclude)
            mean_n, std_n, mean_f, std_f = (
                mean_n[mask],
                std_n[mask],
                mean_f[mask],
                std_f[mask],
            )
            cv_n = cv_n[mask]
            cv_f = cv_f[mask]

        print("\nNormal CV:")
        print(cv_n.round(3))
        print("\nFully CV:")
        print(cv_f.round(3))

        avg_table = pd.concat(
            [mean_n.add_prefix("normal_"), mean_f.add_prefix("fully_")], axis=1
        )
        print("\nAverage Table (ms):")
        print(avg_table.round(3))

        # Save plots to the device folder
        plot_bar_chart(
            mean_n, std_n, mean_f, std_f, device, device_folder, app, backend_col
        )
        diff = mean_f - mean_n
        plot_heatmap(diff, device, device_folder, app, backend_col)
        ratio = mean_f.divide(mean_n)
        plot_ratio_heatmap(ratio, device, device_folder, app, backend_col)
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
