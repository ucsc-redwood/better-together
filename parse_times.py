import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Math model predictions (in seconds)
PREDICTED_TIMES = [
    8.6717,
    8.6717,
    9.07,
    9.1116,
    9.1116,
    9.1116,
    9.1116,
    9.2635,
    9.2635,
    9.5876,
    9.6043,
    9.6043,
    9.6183,
    9.6358,
    9.9833,
    9.9833,
    9.9833,
    9.9833,
    9.9833,
    9.9833,
    9.9833,
    9.9833,
    9.9833,
    9.9833,
    10.1,
    10.1,
    10.3447,
    10.3447,
    10.3447,
    10.3447,
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


def create_visualization(avg_executions, std_executions, predicted_times):
    plt.figure(figsize=(12, 6))

    # Create x-axis values
    x = np.arange(1, len(avg_executions) + 1)

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
    file_path = "accumulated_time.txt"
    runs_data = parse_accumulated_times(file_path)

    # Create a DataFrame where each row is a run and each column is an execution
    df = pd.DataFrame(runs_data)

    # Add row labels
    df.index = [f"Run {i+1}" for i in range(len(runs_data))]

    # Add column labels
    df.columns = [f"Execution {i+1}" for i in range(len(runs_data[0]))]

    # Save individual runs to CSV
    df.to_csv("execution_times.csv")

    # Calculate statistics across all runs for each execution
    avg_executions = df.mean(axis=0)
    std_executions = df.std(axis=0)

    # Calculate percentage error (coefficient of variation)
    percent_error = (std_executions / avg_executions) * 100

    # Use predicted times as is (already in seconds)
    predicted_times = PREDICTED_TIMES

    # Calculate difference between measured and predicted
    diff_times_arith = avg_executions - predicted_times
    diff_percent_arith = (diff_times_arith / predicted_times) * 100

    # Calculate correlation between measured and predicted
    correlation_arith, p_value_arith = stats.pearsonr(avg_executions, predicted_times)

    # Create visualization
    create_visualization(avg_executions, std_executions, predicted_times)

    # Create a DataFrame for the statistics
    stats_df = pd.DataFrame(
        {
            "Measured Avg (s)": avg_executions,
            "Predicted (s)": predicted_times,
            "Diff Arith (s)": diff_times_arith,
            "Diff Arith %": diff_percent_arith,
            "Error %": percent_error,
        }
    )

    # Save statistics to CSV
    stats_df.to_csv("execution_statistics.csv")

    # Print summary statistics
    print(f"Total number of runs: {len(runs_data)}")
    print(f"Executions per run: {len(runs_data[0]) if runs_data else 0}")
    print(f"\nCorrelation between measured and predicted times:")
    print(
        f"Arithmetic mean correlation: {correlation_arith:.4f} (p-value: {p_value_arith:.4f})"
    )
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

    for i, (avg, pred, diff_arith_pct, err_pct) in enumerate(
        zip(
            avg_executions,
            predicted_times,
            diff_percent_arith,
            percent_error,
        ),
        1,
    ):
        print(f"{i:<12} {avg:.4f} {pred:.4f} {diff_arith_pct:+.1f}% {err_pct:.1f}%")


if __name__ == "__main__":
    main()
