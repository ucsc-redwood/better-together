#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_device_data(folder):
    """
    Searches the folder for CSV files following the pattern:
      <device>_normal.csv and <device>_fully.csv.
    Returns a dictionary mapping device IDs to a dict with keys 'normal' and 'fully'
    containing the combined DataFrames from all runs.
    """
    data = {}
    # Look for normal CSV files.
    normal_files = glob.glob(os.path.join(folder, "*_normal.csv"))
    fully_files = glob.glob(os.path.join(folder, "*_fully.csv"))

    for nf in normal_files:
        basename = os.path.basename(nf)
        # Expecting filename like: <device>_normal.csv
        device = basename.split("_normal.csv")[0]
        df = pd.read_csv(nf)
        if device not in data:
            data[device] = {}
        if "normal" in data[device]:
            data[device]["normal"] = pd.concat(
                [data[device]["normal"], df], ignore_index=True
            )
        else:
            data[device]["normal"] = df

    for ff in fully_files:
        basename = os.path.basename(ff)
        device = basename.split("_fully.csv")[0]
        df = pd.read_csv(ff)
        if device not in data:
            data[device] = {}
        if "fully" in data[device]:
            data[device]["fully"] = pd.concat(
                [data[device]["fully"], df], ignore_index=True
            )
        else:
            data[device]["fully"] = df

    return data


def aggregate_data(df, value_cols=["little", "medium", "big", "vulkan"]):
    """
    Group the DataFrame by 'stage' and compute mean and std for the value columns.
    Returns three DataFrames: one with the mean, one with the standard deviation,
    and one with the coefficient of variation (std/mean).
    """
    group = df.groupby("stage")
    mean_df = group[value_cols].mean()
    std_df = group[value_cols].std()
    cv_df = std_df.divide(mean_df).fillna(0)
    return mean_df, std_df, cv_df


def plot_bar_chart(mean_normal, std_normal, mean_fully, std_fully, device, out_folder):
    """
    Create a grouped bar chart showing benchmark performance per stage and PU.
    Two panels are shown: one for normal and one for fully benchmarks.
    Error bars represent the run-to-run variation (std dev).
    Saves the figure as {device}_bar_chart.png in out_folder.
    """
    pu_list = ["little", "medium", "big", "vulkan"]
    stages = mean_normal.index.astype(int).tolist()  # assuming stages are numeric
    x = np.arange(len(stages))  # positions for stages
    width = 0.2  # width of each bar

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
    """
    Plots a heat map of the absolute difference (fully - normal) per stage & PU.
    Rows are stages and columns are processing units.
    Saves the figure as {device}_heatmap.png in out_folder.
    """
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
            text = f"{diff_df.iloc[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    output_path = os.path.join(out_folder, f"{device}_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap for device {device} to: {output_path}")


def plot_normalized_ratio_heatmap(ratio_df, device, out_folder):
    """
    Plots a heat map of the normalized ratio (fully / normal) per stage & PU.
    A value close to 1.0 indicates similar performance (i.e. the core is not affected by system fully overload).
    Saves the heatmap as {device}_ratio_heatmap.png in out_folder.
    """
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
            text = f"{ratio_df.iloc[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    output_path = os.path.join(out_folder, f"{device}_ratio_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved normalized ratio heatmap for device {device} to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark stability, produce average tables and figures, and optionally exclude specified stages."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder path containing the CSV files (and logs) from benchmark runs",
    )
    parser.add_argument(
        "--exclude_stages",
        required=False,
        default="",
        help="Comma-separated list of stage numbers to exclude (e.g., '1,3,7')",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: The folder {args.folder} does not exist.")
        return

    # Parse the exclude stages list if provided.
    exclude_stages = []
    if args.exclude_stages:
        try:
            exclude_stages = [
                int(x.strip()) for x in args.exclude_stages.split(",") if x.strip()
            ]
            print(f"Excluding stages: {exclude_stages}")
        except Exception as e:
            print("Error parsing --exclude_stages argument:", args.exclude_stages, e)
            return

    device_data = load_device_data(args.folder)
    if not device_data:
        print(f"No CSV data found in {args.folder}.")
        return

    for device, datasets in device_data.items():
        print(f"\n=== Analysis for Device: {device} ===")
        if "normal" not in datasets or "fully" not in datasets:
            print(f"Missing benchmark type in data for device {device}. Skipping...")
            continue

        df_normal = datasets["normal"]
        df_fully = datasets["fully"]

        mean_normal, std_normal, cv_normal = aggregate_data(df_normal)
        mean_fully, std_fully, cv_fully = aggregate_data(df_fully)

        # If exclusion is requested, remove those stages from all aggregated DataFrames.
        if exclude_stages:
            mean_normal = mean_normal[
                ~mean_normal.index.astype(int).isin(exclude_stages)
            ]
            std_normal = std_normal[~std_normal.index.astype(int).isin(exclude_stages)]
            cv_normal = cv_normal[~cv_normal.index.astype(int).isin(exclude_stages)]
            mean_fully = mean_fully[~mean_fully.index.astype(int).isin(exclude_stages)]
            std_fully = std_fully[~std_fully.index.astype(int).isin(exclude_stages)]
            cv_fully = cv_fully[~cv_fully.index.astype(int).isin(exclude_stages)]

        print("\nNormal Benchmark Stability (Coefficient of Variation):")
        print(cv_normal.round(3))
        print("\nFully Benchmark Stability (Coefficient of Variation):")
        print(cv_fully.round(3))

        avg_table = pd.concat(
            [mean_normal.add_prefix("normal_"), mean_fully.add_prefix("fully_")], axis=1
        )
        print("\nCombined Average Table (ms per task):")
        print(avg_table.round(3))

        # Generate figures.
        plot_bar_chart(
            mean_normal, std_normal, mean_fully, std_fully, device, args.folder
        )

        diff_df = mean_fully - mean_normal
        plot_heatmap(diff_df, device, args.folder)

        ratio_df = mean_fully.divide(mean_normal)
        plot_normalized_ratio_heatmap(ratio_df, device, args.folder)


if __name__ == "__main__":
    main()
