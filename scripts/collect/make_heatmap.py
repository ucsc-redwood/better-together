#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def aggregate_data(df, value_cols=["little", "medium", "big", "vulkan"]):
    """
    Group the DataFrame by 'stage' and compute mean and std for the value columns.
    Returns three DataFrames: mean, std, cv (coefficient of variation).
    """
    group = df.groupby("stage")
    mean_df = group[value_cols].mean()
    std_df = group[value_cols].std()
    cv_df = std_df.divide(mean_df).fillna(0)
    return mean_df, std_df, cv_df

def plot_bar_chart(mean_normal, std_normal, mean_fully, std_fully, device, out_folder):
    pu_list = ["little", "medium", "big", "vulkan"]
    stages = mean_normal.index.astype(int).tolist()
    x = np.arange(len(stages))
    width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (mean_df, std_df, title) in zip(
        axs,
        [
            (mean_normal, std_normal, "Normal Benchmark"),
            (mean_fully, std_fully, "Fully Benchmark"),
        ],
    ):
        for i, pu in enumerate(pu_list):
            pos = x - (1.5 * width) + i * width
            ax.bar(pos, mean_df[pu], width, yerr=std_df[pu], capsize=5, label=pu)
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_xlabel("Stage")
        ax.set_title(title)
        ax.legend()

    axs[0].set_ylabel("Performance (ms per task)")
    plt.suptitle(f"Average Benchmark Performance for Device {device}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(out_folder, f"{device}_bar_chart.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved bar chart for device {device} to: {output_path}")

def plot_heatmap(diff_df, device, out_folder):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(diff_df.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(diff_df.columns)))
    ax.set_xticklabels(diff_df.columns)
    ax.set_yticks(np.arange(len(diff_df.index)))
    ax.set_yticklabels(diff_df.index.astype(str))
    ax.set_xlabel("Processing Unit")
    ax.set_ylabel("Stage")
    ax.set_title(f"(Fully - Normal) Difference for Device {device}")

    for i in range(diff_df.shape[0]):
        for j in range(diff_df.shape[1]):
            ax.text(j, i, f"{diff_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    output_path = os.path.join(out_folder, f"{device}_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap for device {device} to: {output_path}")

def plot_normalized_ratio_heatmap(ratio_df, device, out_folder):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(ratio_df.values, cmap="coolwarm", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(ratio_df.columns)))
    ax.set_xticklabels(ratio_df.columns)
    ax.set_yticks(np.arange(len(ratio_df.index)))
    ax.set_yticklabels(ratio_df.index.astype(str))
    ax.set_xlabel("Processing Unit")
    ax.set_ylabel("Stage")
    ax.set_title(f"Normalized Ratio (Fully/Normal) for Device {device}")

    for i in range(ratio_df.shape[0]):
        for j in range(ratio_df.shape[1]):
            ax.text(j, i, f"{ratio_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    output_path = os.path.join(out_folder, f"{device}_ratio_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved normalized ratio heatmap for device {device} to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark stability and generate figures for a single device."
    )
    parser.add_argument(
        "--log_folder",
        required=True,
        help="Folder path containing the CSV files from benchmark runs",
    )
    parser.add_argument(
        "--app",
        required=True,
        help="Application name (e.g., cifar-sparse)",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["vk", "cu"],
        help="Backend type: 'vk' or 'cu'",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Device identifier to analyze",
    )
    parser.add_argument(
        "--exclude_stages",
        default="",
        help="Comma-separated list of stage numbers to exclude (e.g., '1,3,7')",
    )
    args = parser.parse_args()

    folder = args.log_folder
    device = args.device

    if not os.path.isdir(folder):
        print(f"Error: The folder {folder} does not exist.")
        return

    # Load CSVs for the given device
    normal_path = os.path.join(folder, f"{device}_normal.csv")
    fully_path = os.path.join(folder, f"{device}_fully.csv")
    if not os.path.exists(normal_path) or not os.path.exists(fully_path):
        print(f"Error: Required CSVs not found for device {device} in {folder}.")
        return
    df_normal = pd.read_csv(normal_path)
    df_fully = pd.read_csv(fully_path)

    # Parse exclude stages
    exclude = []
    if args.exclude_stages:
        try:
            exclude = [int(x) for x in args.exclude_stages.split(',') if x.strip()]
            print(f"Excluding stages: {exclude}")
        except ValueError:
            print(f"Invalid --exclude_stages: {args.exclude_stages}")
            return

    # Aggregate
    mean_n, std_n, cv_n = aggregate_data(df_normal)
    mean_f, std_f, cv_f = aggregate_data(df_fully)

    # Exclude if requested
    if exclude:
        mean_n = mean_n[~mean_n.index.astype(int).isin(exclude)]
        std_n = std_n[~std_n.index.astype(int).isin(exclude)]
        cv_n = cv_n[~cv_n.index.astype(int).isin(exclude)]
        mean_f = mean_f[~mean_f.index.astype(int).isin(exclude)]
        std_f = std_f[~std_f.index.astype(int).isin(exclude)]
        cv_f = cv_f[~cv_f.index.astype(int).isin(exclude)]

    # Print tables
    print("\nNormal Benchmark Stability (Coefficient of Variation):")
    print(cv_n.round(3))
    print("\nFully Benchmark Stability (Coefficient of Variation):")
    print(cv_f.round(3))

    avg_table = pd.concat(
        [mean_n.add_prefix('normal_'), mean_f.add_prefix('fully_')], axis=1
    )
    print("\nCombined Average Table (ms per task):")
    print(avg_table.round(3))

    # Generate and save figures in the log folder
    plot_bar_chart(mean_n, std_n, mean_f, std_f, device, folder)
    diff = mean_f - mean_n
    plot_heatmap(diff, device, folder)
    ratio = mean_f.divide(mean_n)
    plot_normalized_ratio_heatmap(ratio, device, folder)

if __name__ == "__main__":
    main()
