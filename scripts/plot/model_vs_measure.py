import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error

# Schedules to ignore in analysis (by execution number, 1-indexed)
IGNORED_SCHEDULES = [
    # Add execution numbers to ignore here (e.g., 5, 10, 23)
    1,
    2,
    3,
    4,
]

# Math model predictions (in seconds)
PREDICTED_TIMES = [
    121.1783,  # Solution 1
    45.0962,  # Solution 2
    38.908,  # Solution 3
    18.0777,  # Solution 4
    11.9866,  # Solution 5
    9.9833,  # Solution 6
    21.0166,  # Solution 7
    9.9833,  # Solution 8
    11.6786,  # Solution 9
    22.372,  # Solution 10
    9.9833,  # Solution 11
    19.1722,  # Solution 12
    10.1,  # Solution 13
    11.6786,  # Solution 14
    8.6717,  # Solution 15
    10.1,  # Solution 16
    10.6456,  # Solution 17
    17.9516,  # Solution 18
    9.9833,  # Solution 19
    9.07,  # Solution 20
    9.9833,  # Solution 21
    9.9833,  # Solution 22
    36.2851,  # Solution 23
    9.1116,  # Solution 24
    9.2635,  # Solution 25
    9.2635,  # Solution 26
    12.5407,  # Solution 27
    9.1116,  # Solution 28
    10.6373,  # Solution 29
    9.6358,  # Solution 30
    9.6043,  # Solution 31
    9.6043,  # Solution 32
    13.4168,  # Solution 33
    11.6786,  # Solution 34
    31.0949,  # Solution 35
    9.1116,  # Solution 36
    9.9833,  # Solution 37
    11.6786,  # Solution 38
    11.9227,  # Solution 39
    10.3447,  # Solution 40
]


def parse_accumulated_times(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Split content into runs using the separator lines
    runs = content.split("--------------------------------")

    # Remove empty runs and process each run
    runs = [run.strip() for run in runs if run.strip()]

    all_runs_data = []

    for run in runs:
        # Extract all ms values from the run
        ms_values = []
        for line in run.split("\n"):
            if "Total execution time:" in line:
                # Extract the ms value using regex
                match = re.search(r"\((\d+\.\d+) ms\)", line)
                if match:
                    # Convert to seconds by dividing by 100
                    ms_values.append(float(match.group(1)) / 100)

        all_runs_data.append(ms_values)

    return all_runs_data


def create_visualization(
    avg_executions, std_executions, predicted_times, included_indices
):
    plt.figure(figsize=(12, 6))

    # Create x-axis values based on included indices
    x = np.array(included_indices)

    # Plot measured values with error bars
    plt.errorbar(
        x,
        avg_executions,
        yerr=std_executions,
        fmt="o-",
        label="Measured (Arithmetic)",
        capsize=5,
        color="blue",
        alpha=0.7,
    )

    # Plot predicted values
    plt.plot(x, predicted_times, "r--", label="Predicted", linewidth=2)

    # Customize the plot
    plt.xlabel("Execution Number")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Measured vs Predicted Execution Times")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save the plot
    plt.savefig("execution_times_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse execution times from accumulated_time.txt"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="accumulated_time.txt",
        help="Path to input file containing execution times",
    )
    parser.add_argument(
        "--ignore",
        type=int,
        nargs="+",
        help="List of execution numbers to ignore (1-indexed)",
    )
    args = parser.parse_args()

    # Update ignored schedules if provided via command line
    global IGNORED_SCHEDULES
    if args.ignore:
        IGNORED_SCHEDULES.extend(args.ignore)

    file_path = args.input
    runs_data = parse_accumulated_times(file_path)

    # Create a DataFrame where each row is a run and each column is an execution
    df = pd.DataFrame(runs_data)

    # Add row labels
    df.index = [f"Run {i+1}" for i in range(len(runs_data))]

    # Add column labels
    df.columns = [f"Execution {i+1}" for i in range(len(runs_data[0]))]

    # Save individual runs to CSV
    df.to_csv("execution_times.csv")

    # Filter out ignored schedules
    included_columns = [
        i for i in range(len(df.columns)) if i + 1 not in IGNORED_SCHEDULES
    ]
    included_indices = [i + 1 for i in included_columns]

    df_filtered = df.iloc[:, included_columns]

    # Calculate statistics across all runs for each execution
    avg_executions = df_filtered.mean(axis=0)
    std_executions = df_filtered.std(axis=0)

    # Calculate percentage error (coefficient of variation)
    percent_error = (std_executions / avg_executions) * 100

    # Use predicted times for included executions only
    predicted_times_filtered = [
        PREDICTED_TIMES[i] for i in included_columns if i < len(PREDICTED_TIMES)
    ]

    # Ensure we're only considering executions with both measured and predicted values
    valid_count = min(len(avg_executions), len(predicted_times_filtered))
    avg_executions = avg_executions[:valid_count]
    predicted_times_filtered = predicted_times_filtered[:valid_count]
    included_indices = included_indices[:valid_count]

    # Calculate difference between measured and predicted
    diff_times_arith = avg_executions - predicted_times_filtered
    diff_percent_arith = (diff_times_arith / predicted_times_filtered) * 100

    # Calculate correlation between measured and predicted
    correlation_arith, p_value_arith = stats.pearsonr(
        avg_executions, predicted_times_filtered
    )

    # Calculate additional metrics
    mse = mean_squared_error(avg_executions, predicted_times_filtered)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff_times_arith))
    r_squared = correlation_arith**2

    # Create visualization with filtered data
    create_visualization(
        avg_executions,
        std_executions[:valid_count],
        predicted_times_filtered,
        included_indices,
    )

    # Create a DataFrame for the statistics
    stats_df = pd.DataFrame(
        {
            "Execution": included_indices,
            "Measured Avg (s)": avg_executions,
            "Predicted (s)": predicted_times_filtered,
            "Diff Arith (s)": diff_times_arith,
            "Diff Arith %": diff_percent_arith,
            "Error %": percent_error[:valid_count],
        }
    )

    # Save statistics to CSV
    stats_df.to_csv("execution_statistics.csv", index=False)

    # Print summary statistics
    print(f"Total number of runs: {len(runs_data)}")
    print(f"Executions per run: {len(runs_data[0]) if runs_data else 0}")
    print(
        f"Executions included in analysis: {valid_count} (ignored: {sorted(IGNORED_SCHEDULES)})"
    )

    print(f"\nCorrelation and Error Metrics:")
    print(
        f"Pearson correlation: {correlation_arith:.4f} (p-value: {p_value_arith:.4f})"
    )
    print(f"R-squared: {r_squared:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    print("\nIndividual runs data has been saved to 'execution_times.csv'")
    print("Execution statistics have been saved to 'execution_statistics.csv'")
    print("Visualization has been saved to 'execution_times_comparison.png'")

    # Print the statistics in a formatted table
    print("\nExecution Statistics (Measured vs Predicted):")
    print("=" * 100)
    print(
        f"{'Execution':<12} {'Arith (s)':<12} {'Predicted (s)':<12} {'Diff Arith %':<12} {'Error %':<12}"
    )
    print("-" * 100)

    for i, exec_num, avg, pred, diff_arith_pct, err_pct in zip(
        range(len(included_indices)),
        included_indices,
        avg_executions,
        predicted_times_filtered,
        diff_percent_arith,
        percent_error[:valid_count],
    ):
        print(
            f"{exec_num:<12} {avg:.4f} {pred:.4f} {diff_arith_pct:+.1f}% {err_pct:.1f}%"
        )


if __name__ == "__main__":
    main()
