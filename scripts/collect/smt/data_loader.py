"""Data loading and processing utilities for schedule optimization."""

import pandas as pd
from .baselines import get_num_stages_for_app


def load_csv_and_compute_averages(csv_path, app_name, verbose=False):
    """
    Load data from a CSV file and compute average timings for each stage across all runs.
    The CSV input contains multiple runs for the same stage. like this:
        stage,little,medium,big,vulkan,cuda,device,run
            1,...
            2,...
            ...
            7,...
            8,...
            9,...
            1,...
            2,...
            3,...
            4,...
            ...
    We want to merge the runs for the same stage and compute the average.

    Args:
        csv_path: Path to the CSV file containing stage timing data
        app_name: Name of the application to determine number of stages

    Returns:
        A tuple of (avg_timings, use_cuda) where avg_timings is a list of lists
        containing average timing data for each stage and core type
    """
    # Load CSV file
    df = pd.read_csv(csv_path)

    # Compute average for each stage across all runs
    # Group by stage and calculate mean for each core type
    avg_df = df.groupby("stage")[["little", "medium", "big", "vulkan", "cuda"]].mean()

    # Print the average table
    if verbose:
        print("\n=== Average Stage Timings ===")
        print(avg_df)
        print()

    # Determine which GPU backend to use (CUDA or Vulkan)
    # If CUDA values are all zeros, use Vulkan
    # Otherwise, use CUDA as the GPU backend
    use_cuda = avg_df["cuda"].sum() > 0

    if verbose:
        print(f"Using {'CUDA' if use_cuda else 'Vulkan'} as the GPU backend")

    # Get application-specific stage count
    num_stages = get_num_stages_for_app(app_name) if app_name else 9

    # Convert to list of lists format expected by the solver
    # Each inner list is [little_time, medium_time, big_time, gpu_time]
    # where gpu_time is either vulkan_time or cuda_time
    avg_timings = []
    for stage in range(1, num_stages + 1):  # Use app-specific number of stages
        if stage in avg_df.index:
            row = avg_df.loc[stage]
            # Use cuda if available, otherwise use vulkan
            gpu_time = row["cuda"] if use_cuda else row["vulkan"]
            avg_timings.append([row["little"], row["medium"], row["big"], gpu_time])
        else:
            print(f"Warning: Stage {stage} not found in data")
            # Use zeros as fallback
            avg_timings.append([0.0, 0.0, 0.0, 0.0])

    return avg_timings, use_cuda


def define_data(stage_timings=None, app_name=None):
    """Define the problem data."""
    # Get application-specific stage count if available
    num_stages = get_num_stages_for_app(app_name) if app_name else 9
    core_types = ["Little", "Medium", "Big", "GPU"]

    # Use provided stage timings if available, otherwise use default values
    if stage_timings is not None:
        return num_stages, core_types, stage_timings

    # Default timings if no CSV is provided
    default_stage_timings = []

    return num_stages, core_types, default_stage_timings
