#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set the style for the plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Define the path to the data and figure directories
DATA_DIR = 'data/bm_logs'
FIG_DIR = 'data/figures'

# Create the figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

def get_device_app_backend_type_from_filename(filename):
    """Extract device, application, backend, and type from filename."""
    basename = os.path.basename(filename)
    parts = basename.replace('.csv', '').split('_')
    
    if len(parts) != 4:
        return None, None, None, None
    
    device, app, backend, run_type = parts
    return device, app, backend, run_type

def get_gpu_column_for_device(device):
    """Return the appropriate GPU column name based on the device."""
    if 'jetson' in device.lower():
        return 'cuda'
    else:
        return 'vulkan'

def read_and_aggregate_data():
    """Read all CSV files and aggregate the data."""
    all_data = []
    
    # Get all CSV files in the data directory
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    for file_path in csv_files:
        device, app, backend, run_type = get_device_app_backend_type_from_filename(file_path)
        
        if device is None:
            continue
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Group by stage and calculate mean for each processor
        grouped = df.groupby('stage').agg({
            'little': 'mean',
            'medium': 'mean',
            'big': 'mean',
            'vulkan': 'mean',
            'cuda': 'mean'
        }).reset_index()
        
        # Add device, app, backend, and run_type columns
        grouped['device'] = device
        grouped['app'] = app
        grouped['backend'] = backend
        grouped['run_type'] = run_type
        
        # Add a gpu column that uses cuda for jetson devices, vulkan for others
        gpu_col = get_gpu_column_for_device(device)
        grouped['gpu'] = grouped[gpu_col]
        
        all_data.append(grouped)
    
    # Combine all data
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_overview_figures(data):
    """Create overview figures grouped by device and application."""
    # Get unique devices and applications
    devices = data['device'].unique()
    applications = data['app'].unique()
    
    # Create a figure for each combination of device and run_type
    for run_type in ['fully', 'normal']:
        # Filter data for the current run_type
        run_type_data = data[data['run_type'] == run_type]
        
        if run_type_data.empty:
            continue
        
        # Create a figure with 3x3 subplots (or adjust based on number of devices/apps)
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(len(devices), len(applications), figure=fig)
        
        for i, device in enumerate(devices):
            for j, app in enumerate(applications):
                # Filter data for current device and app
                subset = run_type_data[(run_type_data['device'] == device) & 
                                     (run_type_data['app'] == app)]
                
                if subset.empty:
                    continue
                
                # Create subplot
                ax = fig.add_subplot(gs[i, j])
                
                # Prepare data for plotting
                processors = ['little', 'medium', 'big', 'gpu']
                
                # Create a plot for each stage
                for stage in subset['stage'].unique():
                    stage_data = subset[subset['stage'] == stage]
                    
                    if stage_data.empty:
                        continue
                    
                    # For each processor, get the first entry for this stage
                    proc_values = [stage_data[proc].values[0] for proc in processors]
                    
                    # Use a logarithmic scale for better visibility
                    ax.bar(range(len(processors)), proc_values, label=f'Stage {int(stage)}')
                
                # Set title and labels
                ax.set_title(f'{app} on {device}')
                ax.set_xticks(range(len(processors)))
                ax.set_xticklabels(processors, rotation=45)
                ax.set_yscale('log')
                ax.set_ylabel('Time (ms) - Log Scale')
                
                # Add a legend outside the plot
                if i == 0 and j == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'overview_{run_type}.png'))
        plt.close()

def create_heatmap(data):
    """Create heatmaps showing the performance difference between processors."""
    for run_type in ['fully', 'normal']:
        run_type_data = data[data['run_type'] == run_type]
        
        if run_type_data.empty:
            continue
        
        # Create a figure with 2 rows (one for each device)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        devices = run_type_data['device'].unique()
        applications = run_type_data['app'].unique()
        
        for i, device in enumerate(devices[:2]):  # Limit to 2 devices for the 2x3 grid
            for j, app in enumerate(applications[:3]):  # Limit to 3 apps for the 2x3 grid
                subset = run_type_data[(run_type_data['device'] == device) & 
                                     (run_type_data['app'] == app)]
                
                if subset.empty:
                    continue
                
                # Create a matrix of processor performance ratios
                processors = ['little', 'medium', 'big', 'gpu']
                ratio_matrix = np.zeros((len(processors), len(processors)))
                
                # Calculate average speedups across all stages
                for stage in subset['stage'].unique():
                    stage_data = subset[subset['stage'] == stage]
                    
                    if stage_data.empty:
                        continue
                    
                    # For each pair of processors, calculate speedup
                    for p1_idx, p1 in enumerate(processors):
                        for p2_idx, p2 in enumerate(processors):
                            # Skip diagonal
                            if p1 == p2:
                                continue
                                
                            # Calculate speedup: p1 time / p2 time
                            # If p1 is slower than p2, ratio > 1 (positive speedup)
                            # If p1 is faster than p2, ratio < 1 (negative speedup)
                            if stage_data[p2].values[0] == 0:
                                continue  # Avoid division by zero
                                
                            ratio = stage_data[p1].values[0] / stage_data[p2].values[0]
                            ratio_matrix[p1_idx, p2_idx] += ratio
                
                # Normalize by number of stages
                num_stages = len(subset['stage'].unique())
                ratio_matrix /= num_stages
                
                # Set diagonal to 1 (same processor)
                for k in range(len(processors)):
                    ratio_matrix[k, k] = 1
                
                # Convert to log scale for better visualization
                log_ratio = np.log2(ratio_matrix)
                
                # Create heatmap
                ax = axes[i, j]
                sns.heatmap(log_ratio, annot=True, fmt=".2f", cmap="RdBu_r", 
                           xticklabels=processors, yticklabels=processors, ax=ax,
                           vmin=-4, vmax=4, center=0)
                
                ax.set_title(f'{app} on {device}')
                
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'heatmap_{run_type}.png'))
        plt.close()

def create_grouped_bar_plots(data):
    """Create grouped bar plots showing stage execution times across processors."""
    for run_type in ['fully', 'normal']:
        run_type_data = data[data['run_type'] == run_type]
        
        if run_type_data.empty:
            continue
        
        devices = run_type_data['device'].unique()
        applications = run_type_data['app'].unique()
        
        for device in devices:
            for app in applications:
                subset = run_type_data[(run_type_data['device'] == device) & 
                                     (run_type_data['app'] == app)]
                
                if subset.empty:
                    continue
                
                # Get the stages for this app
                stages = sorted(subset['stage'].unique())
                
                # Create a figure
                fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
                
                # Set width of bars
                bar_width = 0.2
                
                # Set position of bars on x axis
                r1 = np.arange(len(stages))
                r2 = [x + bar_width for x in r1]
                r3 = [x + bar_width for x in r2]
                r4 = [x + bar_width for x in r3]
                
                # Make the plot
                processors = ['little', 'medium', 'big', 'gpu']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                
                # For each processor, plot the stage times
                for i, processor in enumerate(processors):
                    values = [subset[subset['stage'] == stage][processor].values[0] for stage in stages]
                    positions = [r1, r2, r3, r4][i]
                    ax.bar(positions, values, width=bar_width, color=colors[i], label=processor)
                
                # Add labels and legend
                ax.set_xlabel('Stage')
                ax.set_ylabel('Time (ms) - Log Scale')
                ax.set_title(f'Stage Execution Times for {app} on {device} ({run_type})')
                ax.set_xticks([r + bar_width * 1.5 for r in range(len(stages))])
                ax.set_xticklabels([f'Stage {int(s)}' for s in stages])
                ax.set_yscale('log')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, f'stages_{device}_{app}_{run_type}.png'))
                plt.close()

def create_device_comparison_plots(data):
    """Create plots comparing performance between devices for the same application."""
    for run_type in ['fully', 'normal']:
        run_type_data = data[data['run_type'] == run_type]
        
        if run_type_data.empty:
            continue
        
        devices = run_type_data['device'].unique()
        applications = run_type_data['app'].unique()
        processors = ['little', 'medium', 'big', 'gpu']
        
        for app in applications:
            # Create a figure with subplots for each processor
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for p_idx, processor in enumerate(processors):
                ax = axes[p_idx]
                
                # For each device, get the stage times for this processor
                for device in devices:
                    subset = run_type_data[(run_type_data['device'] == device) & 
                                        (run_type_data['app'] == app)]
                    
                    if subset.empty:
                        continue
                    
                    # Get the stages for this app
                    stages = sorted(subset['stage'].unique())
                    
                    # Get the times for each stage
                    times = [subset[subset['stage'] == stage][processor].values[0] for stage in stages]
                    
                    # Plot the times
                    ax.plot(stages, times, marker='o', label=device)
                
                # Add labels and legend
                ax.set_xlabel('Stage')
                ax.set_ylabel('Time (ms) - Log Scale')
                ax.set_title(f'{processor} Performance for {app} ({run_type})')
                ax.set_yscale('log')
                ax.grid(True)
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f'device_comparison_{app}_{run_type}.png'))
            plt.close()

def create_summary_table(data):
    """Create a summary table comparing performance across devices and applications."""
    # Create a summary DataFrame
    summary_rows = []
    
    # Calculate mean performance for each processor across all stages
    for device in data['device'].unique():
        for app in data['app'].unique():
            for run_type in data['run_type'].unique():
                subset = data[(data['device'] == device) & 
                             (data['app'] == app) & 
                             (data['run_type'] == run_type)]
                
                if subset.empty:
                    continue
                
                # Calculate mean for each processor
                for processor in ['little', 'medium', 'big', 'gpu']:
                    mean_time = subset[processor].mean()
                    
                    # Add to summary rows list
                    summary_rows.append({
                        'device': device,
                        'app': app,
                        'run_type': run_type,
                        'processor': processor,
                        'mean_time': mean_time
                    })
    
    # Create DataFrame from rows
    summary = pd.DataFrame(summary_rows)
    
    # Save the summary table to CSV
    summary.to_csv(os.path.join(FIG_DIR, 'performance_summary.csv'), index=False)
    
    # Create a pivot table for easier comparison
    pivot = summary.pivot_table(
        index=['app', 'run_type'],
        columns=['device', 'processor'],
        values='mean_time'
    )
    
    # Save the pivot table to CSV
    pivot.to_csv(os.path.join(FIG_DIR, 'performance_pivot.csv'))
    
    return summary

def main():
    # Read and aggregate the data
    data = read_and_aggregate_data()
    
    if data.empty:
        print("No data found. Please check the path to the CSV files.")
        return
    
    # Create overview figures
    create_overview_figures(data)
    
    # Create heatmaps
    create_heatmap(data)
    
    # Create grouped bar plots
    create_grouped_bar_plots(data)
    
    # Create device comparison plots
    create_device_comparison_plots(data)
    
    # Create summary table
    create_summary_table(data)
    
    print(f"Report generated successfully. Figures saved to {FIG_DIR}")

if __name__ == "__main__":
    main() 