import re
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def parse_benchmark_data(raw_data: str) -> pd.DataFrame:
    """Parse benchmark data with 4 lines per stage into a structured DataFrame."""
    pattern = re.compile(r"(\w+)=([^|]+)")
    records = []

    lines = raw_data.strip().splitlines()

    # Calculate the number of processors per stage by looking for blank lines or
    # using 4 as default if no blank lines
    if "" in lines:
        processors_per_stage = lines.index("")
    else:
        processors_per_stage = 4  # Default assumption: 4 processors per stage

    # Remove any blank lines that might exist in the data
    lines = [line for line in lines if line.strip()]

    # Make sure the data is properly formatted
    assert (
        len(lines) % processors_per_stage == 0
    ), f"Expected {processors_per_stage} lines per stage."

    for i, line in enumerate(lines):
        stage = i // processors_per_stage  # Integer division
        parsed = dict(re.findall(pattern, line))
        parsed["Stage"] = stage
        records.append(parsed)

    df = pd.DataFrame(records)

    # Convert columns to proper types
    for col in df.columns:
        if col not in ["PROCESSOR"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def analyze_benchmark_data(df: pd.DataFrame, output_path: str = None):
    """Analyze and optionally visualize the benchmark data."""
    # Display basic statistics
    print("\nBasic statistics per processor and stage:")
    pivot_avg = df.pivot_table(
        values="AVG", index="Stage", columns="PROCESSOR", aggfunc="first"
    )
    print(pivot_avg)

    # Display min/max statistics
    print("\nMin/Max values per processor:")
    for processor in df["PROCESSOR"].unique():
        subset = df[df["PROCESSOR"] == processor]
        print(f"\n{processor}:")
        print(
            f"  Min AVG: {subset['AVG'].min():.4f} (Stage {subset.loc[subset['AVG'].idxmin(), 'Stage']})"
        )
        print(
            f"  Max AVG: {subset['AVG'].max():.4f} (Stage {subset.loc[subset['AVG'].idxmax(), 'Stage']})"
        )
        print(f"  Overall MIN: {subset['MIN'].min():.4f}")
        print(f"  Overall MAX: {subset['MAX'].max():.4f}")

    # Create visualization if requested
    if output_path:
        # Create a bar chart instead of a line chart for independent stages
        plt.figure(figsize=(14, 10))

        # Convert data to long format for easier plotting
        df_plot = df.copy()
        df_plot["Stage"] = df_plot["Stage"].astype(
            str
        )  # Convert to string for categorical x-axis

        # Create the bar chart
        ax = sns.barplot(data=df_plot, x="Stage", y="AVG", hue="PROCESSOR")

        plt.title("Average Performance by Processor and Stage", fontsize=14)
        plt.xlabel("Stage", fontsize=12)
        plt.ylabel("Average Time (ms)", fontsize=12)
        plt.grid(True, alpha=0.3, axis="y")
        plt.legend(title="Processor")

        # Adjust xticks
        plt.xticks(rotation=0)

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"\nBar chart visualization saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python parse_full_benchmark.py <benchmark_file> [output_image.png]"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_file, "r") as f:
        raw_input = f.read()

    df = parse_benchmark_data(raw_input)

    # Save to CSV
    csv_output = input_file.split("/")[-1]
    if csv_output.endswith(".txt.tmp"):
        csv_output = f"parsed_{csv_output.replace('.txt.tmp', '')}.csv"
    else:
        csv_output = f"parsed_{csv_output}.csv"

    df.to_csv(csv_output, index=False)
    print(f"Saved parsed benchmark to '{csv_output}'")

    # Display the first few rows with columns we know exist
    print("\nFirst few rows of the parsed data:")
    # Get a list of columns that definitely exist in our data
    display_cols = ["PROCESSOR", "AVG", "MIN", "MAX", "Stage"]
    # Filter to only include columns that exist in the DataFrame
    existing_cols = [col for col in display_cols if col in df.columns]
    print(df[existing_cols].head())

    # Analyze the data
    analyze_benchmark_data(df, output_file)
