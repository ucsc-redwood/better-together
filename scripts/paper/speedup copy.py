
baselines = {
    "3A021JEHN02756": {
        "cifar-dense-vk": {
            "omp": 940,
            "vk": 11.4,
        },
        "cifar-sparse-vk": {
            "omp": 45.8,
            "vk": 44.9,
        },
        "tree-vk": {
            "omp": 14.2,
            "vk": 58.7,
        },
    },
    "9b034f1b": {
        "cifar-dense-vk": {
            "omp": 730,
            "vk": 12.1,
        },
        "cifar-sparse-vk": {
            "omp": 53.2,
            "vk": 27.9,
        },
        "tree-vk": {
            "omp": 12.7,
            "vk": 47.2,
        },
    },
    "jetson": {
        "cifar-dense-cu": {
            "omp": 23.5,
            "cu": 5.48,
        },
        "cifar-sparse-cu": {
            "omp": 486,
            "cu": 27.2,
        },
        "tree-cu": {
            "omp": 16.2,
            "cu": 5.42,
        },
    },
    "jetsonlowpower": {
        "cifar-dense-cu": {
            "omp": 58.5,
            "cu": 23.6,
        },
        "cifar-sparse-cu": {
            "omp": 1042,
            "cu": 101,
        },
        "tree-cu": {
            "omp": 39.7,
            "cu": 7.28,
        },
    },
}


cifar_sparse = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-B1G3L1M4-G242-91e0         :         5.34            7.65          -30.18%
SCH-M1G3L1B4-G197-d983         :         5.38            7.86          -31.57%
SCH-M1G3B3L2-G221-aa43         :         4.23            7.86          -46.15%
SCH-B1G3M3L2-G264-55fd         :         3.96            7.86          -49.69%
SCH-G3M2B4-G208-03eb           :         7.67            9.95          -22.88%
SCH-G3B2M4-G277-491e           :         5.35            9.95          -46.18%
SCH-G3B1M5-G281-0d37           :         6.99            9.95          -29.77%
SCH-G3B1L1M4-G281-92d5         :         5.48            9.95          -44.91%
SCH-G3M1L1B4-G300-f8c0         :         5.86            9.95          -41.11%
SCH-G3M1B5-G300-924d           :         7.37            9.95          -25.95%
SCH-G4B5-G223-b4d8             :         8.38           11.95          -29.87%
SCH-G6L3-G046-d2bb             :        15.17           15.74           -3.61%
SCH-G9-G000-d386               :        33.44           19.39          +72.49%
SCH-B2M7-G111-f30e             :        15.01           19.48          -22.97%
SCH-M2B7-G067-d987             :        14.12           20.00          -29.43%
SCH-M4L5-G029-4af3             :        21.79           30.17          -27.79%
SCH-B4L5-G152-935b             :        22.17           30.17          -26.51%
SCH-B9-G000-c3e9               :        26.72           38.38          -30.38%
SCH-M9-G000-d2cb               :        30.19           38.81          -22.21%
SCH-L9-G000-7d07               :        68.61          108.77          -36.92%
"""

cifar_dense = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G7L2-G195-0c02             :         6.20            6.54           -5.14%
SCH-G7B2-G499-110a             :         6.04            6.54           -7.53%
SCH-G7M2-G503-02ca             :         6.27            6.54           -4.00%
SCH-G7B1L1-G559-3f12           :         6.14            6.54           -6.03%
SCH-G7M1L1-G564-a5c2           :         5.49            6.54          -15.96%
SCH-G7L1M1-G593-3e59           :         5.57            6.54          -14.74%
SCH-G7B1M1-G593-5996           :         6.14            6.54           -6.06%
SCH-G7L1B1-G593-34a5           :         6.42            6.54           -1.79%
SCH-G7M1B1-G593-ba5e           :         6.00            6.54           -8.13%
SCH-G8L1-G553-ec03             :         6.63            6.78           -2.16%
SCH-G8M1-G617-1052             :         6.72            6.78           -0.83%
SCH-G8B1-G617-1601             :         6.56            6.78           -3.19%
SCH-G9-G000-de67               :         9.55            7.54          +26.59%
SCH-B1G8-G5149-0ef5            :         7.27           58.31          -87.52%
SCH-B1L1G7-G5184-812c          :         6.56           58.31          -88.76%
SCH-B1G6L2-G5372-1b82          :         5.80           58.31          -90.05%
SCH-M6L3-G1769-caf2            :       538.67          718.34          -25.01%
SCH-M9-G000-027e               :       793.40         1067.89          -25.70%
SCH-B9-G000-0f7f               :       895.68         1136.26          -21.17%
SCH-L9-G000-416a               :      2008.69         2388.01          -15.88%
"""

tree = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-B2G2L2M1-G042-fedd         :         2.05            2.41          -15.16%
SCH-M2G2L2B1-G045-683f         :         1.69            2.54          -33.53%
SCH-L1M2G3B1-G054-957a         :         0.76            2.62          -71.11%
SCH-L1B2G3M1-G065-ff79         :         0.87            2.62          -66.94%
SCH-L1M2G2B2-G074-dae0         :         1.52            2.62          -41.96%
SCH-L1B2G2M2-G074-b58e         :         1.81            2.62          -30.77%
SCH-B3G3M1-G066-7162           :         1.38            2.66          -48.05%
SCH-M2G2B3-G038-90df           :         2.33            2.80          -16.56%
SCH-M2L1G3B1-G077-1eb3         :         1.07            2.85          -62.55%
SCH-B2G2M3-G054-076e           :         2.64            2.86           -7.58%
SCH-M2G3B2-G045-70c0           :         2.41            2.98          -19.23%
SCH-B2G3M2-G066-fe33           :         3.33            2.98          +11.47%
SCH-B4L3-G029-6a78             :         5.26            8.24          -36.25%
SCH-M4L3-G004-dff0             :         6.33            8.29          -23.67%
SCH-L3M4-G058-b1fa             :         3.80            8.82          -56.88%
SCH-L3B4-G073-d1c7             :         3.63            8.82          -58.82%
SCH-B7-G000-5c06               :         6.92           10.75          -35.65%
SCH-M7-G000-02e4               :         8.34           11.14          -25.12%
SCH-L7-G000-ad72               :        18.06           26.52          -31.91%
SCH-G7-G000-d1bd               :        55.01           27.45         +100.38%
"""


import pandas as pd
import io
import numpy as np
from scipy import stats


def parse_speedup_data(data_string):
    # Skip the header lines and parse the data
    lines = data_string.strip().split("\n")[4:]  # Skip the first 4 lines

    # Create lists to store the data
    schedules = []
    measured_times = []
    predicted_times = []
    differences = []

    for line in lines:
        parts = line.split(":")
        if len(parts) < 2:
            continue

        schedule = parts[0].strip()
        time_parts = parts[1].strip().split()

        if len(time_parts) >= 3:
            measured = float(time_parts[0])
            predicted = float(time_parts[1])
            difference = time_parts[2]

            schedules.append(schedule)
            measured_times.append(measured)
            predicted_times.append(predicted)
            differences.append(difference)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Schedule": schedules,
            "Measured (ms)": measured_times,
            "Predicted (ms)": predicted_times,
            "Difference (%)": differences,
        }
    )

    return df


def compute_geomean_speedup(df):
    # Calculate speedup ratios (predicted/measured)
    speedups = df["Predicted (ms)"] / df["Measured (ms)"]
    # Compute geometric mean
    geomean = np.exp(np.mean(np.log(speedups)))
    return geomean


def analyze_statistics(df):
    # Calculate speedup ratios
    speedups = df["Predicted (ms)"] / df["Measured (ms)"]
    
    # Basic statistics for measured times
    measured_stats = {
        "Mean": np.mean(df["Measured (ms)"]),
        "Median": np.median(df["Measured (ms)"]),
        "Std Dev": np.std(df["Measured (ms)"]),
        "Min": np.min(df["Measured (ms)"]),
        "Max": np.max(df["Measured (ms)"]),
        "25th Percentile": np.percentile(df["Measured (ms)"], 25),
        "75th Percentile": np.percentile(df["Measured (ms)"], 75),
    }
    
    # Basic statistics for predicted times
    predicted_stats = {
        "Mean": np.mean(df["Predicted (ms)"]),
        "Median": np.median(df["Predicted (ms)"]),
        "Std Dev": np.std(df["Predicted (ms)"]),
        "Min": np.min(df["Predicted (ms)"]),
        "Max": np.max(df["Predicted (ms)"]),
        "25th Percentile": np.percentile(df["Predicted (ms)"], 25),
        "75th Percentile": np.percentile(df["Predicted (ms)"], 75),
    }
    
    # Speedup statistics
    speedup_stats = {
        "Geometric Mean": np.exp(np.mean(np.log(speedups))),
        "Arithmetic Mean": np.mean(speedups),
        "Median": np.median(speedups),
        "Std Dev": np.std(speedups),
        "Min": np.min(speedups),
        "Max": np.max(speedups),
        "25th Percentile": np.percentile(speedups, 25),
        "75th Percentile": np.percentile(speedups, 75),
    }
    
    # Correlation between measured and predicted times
    correlation = stats.pearsonr(df["Measured (ms)"], df["Predicted (ms)"])[0]
    
    return {
        "Measured Times": measured_stats,
        "Predicted Times": predicted_stats,
        "Speedup Ratios": speedup_stats,
        "Correlation": correlation
    }


def print_statistics(stats, dataset_name):
    print(f"\n=== {dataset_name} Statistics ===")
    
    print("\nMeasured Times (ms):")
    for stat, value in stats["Measured Times"].items():
        print(f"{stat}: {value:.2f}")
    
    print("\nPredicted Times (ms):")
    for stat, value in stats["Predicted Times"].items():
        print(f"{stat}: {value:.2f}")
    
    print("\nSpeedup Ratios (Predicted/Measured):")
    for stat, value in stats["Speedup Ratios"].items():
        print(f"{stat}: {value:.2f}x")
    
    print(f"\nCorrelation between Measured and Predicted Times: {stats['Correlation']:.3f}")


# Parse each dataset
cifar_sparse_df = parse_speedup_data(cifar_sparse)
cifar_dense_df = parse_speedup_data(cifar_dense)
tree_df = parse_speedup_data(tree)

# Compute statistics for each dataset
cifar_sparse_stats = analyze_statistics(cifar_sparse_df)
cifar_dense_stats = analyze_statistics(cifar_dense_df)
tree_stats = analyze_statistics(tree_df)

# Print results for each application
print("CIFAR Sparse Dataset:")
print(cifar_sparse_df[["Schedule", "Measured (ms)", "Predicted (ms)"]])
print_statistics(cifar_sparse_stats, "CIFAR Sparse")

print("\nCIFAR Dense Dataset:")
print(cifar_dense_df[["Schedule", "Measured (ms)", "Predicted (ms)"]])
print_statistics(cifar_dense_stats, "CIFAR Dense")

print("\nTree Dataset:")
print(tree_df[["Schedule", "Measured (ms)", "Predicted (ms)"]])
print_statistics(tree_stats, "Tree")
