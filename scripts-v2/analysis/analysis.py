#!/usr/bin/env python3

import os
import re
import sys
import glob
import argparse
import numpy as np
from collections import defaultdict

def parse_benchmark_file(file_path):
    """Parse a benchmark file to extract task and chunk duration information."""
    tasks = []
    current_task = None
    current_chunk = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Task start
            if line.startswith('Task '):
                if current_task is not None:
                    tasks.append(current_task)
                task_id = int(line.split(' ')[1].strip(':'))
                current_task = {'id': task_id, 'chunks': []}
            
            # Chunk start
            elif line.startswith('  Chunk '):
                current_chunk = {'id': int(line.split(' ')[2].strip(':')), 'start': 0, 'end': 0, 'duration': 0}
            
            # Start time
            elif line.startswith('    Start: '):
                if current_chunk is not None:
                    current_chunk['start'] = int(line.split(' ')[1])
            
            # End time
            elif line.startswith('    End: '):
                if current_chunk is not None:
                    current_chunk['end'] = int(line.split(' ')[1])
            
            # Duration
            elif line.startswith('    Duration: '):
                if current_chunk is not None:
                    current_chunk['duration'] = int(line.split(' ')[1])
                    current_task['chunks'].append(current_chunk)
                    current_chunk = None
    
    # Add the last task if there is one
    if current_task is not None:
        tasks.append(current_task)
    
    return tasks

def calculate_statistics(tasks):
    """Calculate various statistics from the task data."""
    if not tasks:
        return None
    
    stats = {
        'task_count': len(tasks),
        'chunk_count_per_task': len(tasks[0]['chunks']),
        'chunk_stats': [],
        'task_duration_stats': {},
        'widest_chunk': {'id': -1, 'avg_duration': 0},
        'widest_task': {'id': -1, 'duration': 0},
        'chunk_durations': defaultdict(list),
        'task_durations': []
    }
    
    # Calculate chunk statistics
    for chunk_idx in range(stats['chunk_count_per_task']):
        durations = [task['chunks'][chunk_idx]['duration'] for task in tasks if chunk_idx < len(task['chunks'])]
        stats['chunk_durations'][chunk_idx] = durations
        
        chunk_stats = {
            'id': chunk_idx,
            'min': np.min(durations),
            'max': np.max(durations),
            'mean': np.mean(durations),
            'median': np.median(durations),
            'geo_mean': np.exp(np.mean(np.log(durations))),
            'std_dev': np.std(durations),
            'total_durations': np.sum(durations)
        }
        stats['chunk_stats'].append(chunk_stats)
        
        # Check if this is the widest chunk on average
        if chunk_stats['mean'] > stats['widest_chunk']['avg_duration']:
            stats['widest_chunk'] = {'id': chunk_idx, 'avg_duration': chunk_stats['mean']}
    
    # Calculate task durations (end of last chunk - start of first chunk)
    for task in tasks:
        if len(task['chunks']) > 0:
            start_time = task['chunks'][0]['start']
            end_time = task['chunks'][-1]['end']
            duration = end_time - start_time
            stats['task_durations'].append(duration)
            
            # Check if this is the widest task
            if duration > stats['widest_task']['duration']:
                stats['widest_task'] = {'id': task['id'], 'duration': duration}
    
    # Calculate task duration statistics
    task_durations = stats['task_durations']
    stats['task_duration_stats'] = {
        'min': np.min(task_durations),
        'max': np.max(task_durations),
        'mean': np.mean(task_durations),
        'median': np.median(task_durations),
        'geo_mean': np.exp(np.mean(np.log(task_durations))),
        'std_dev': np.std(task_durations),
        'total_durations': np.sum(task_durations)
    }
    
    return stats

def format_duration(duration_us):
    """Format duration in microseconds to a readable format."""
    if duration_us < 1000:
        return f"{duration_us:.2f} µs"
    elif duration_us < 1000000:
        return f"{duration_us/1000:.2f} ms"
    else:
        return f"{duration_us/1000000:.2f} s"

def print_statistics(file_name, stats):
    """Print statistics in a readable format."""
    print(f"\n{'='*80}")
    print(f"Statistics for {file_name}")
    print(f"{'='*80}")
    
    print(f"\nTask Count: {stats['task_count']}")
    print(f"Chunks per Task: {stats['chunk_count_per_task']}")
    
    print("\nChunk Statistics:")
    print(f"{'Chunk ID':<10}{'Min':<15}{'Max':<15}{'Mean':<15}{'Median':<15}{'Geo Mean':<15}{'Std Dev':<15}")
    print(f"{'-'*80}")
    for chunk in stats['chunk_stats']:
        print(f"{chunk['id']:<10}"
              f"{format_duration(chunk['min']):<15}"
              f"{format_duration(chunk['max']):<15}"
              f"{format_duration(chunk['mean']):<15}"
              f"{format_duration(chunk['median']):<15}"
              f"{format_duration(chunk['geo_mean']):<15}"
              f"{format_duration(chunk['std_dev']):<15}")
    
    print("\nTask Duration Statistics:")
    task_stats = stats['task_duration_stats']
    print(f"Min: {format_duration(task_stats['min'])}")
    print(f"Max: {format_duration(task_stats['max'])}")
    print(f"Mean: {format_duration(task_stats['mean'])}")
    print(f"Median: {format_duration(task_stats['median'])}")
    print(f"Geometric Mean: {format_duration(task_stats['geo_mean'])}")
    print(f"Standard Deviation: {format_duration(task_stats['std_dev'])}")
    
    print(f"\nWidest Chunk: Chunk {stats['widest_chunk']['id']} "
          f"(Average Duration: {format_duration(stats['widest_chunk']['avg_duration'])})")
    
    print(f"Widest Task: Task {stats['widest_task']['id']} "
          f"(Duration: {format_duration(stats['widest_task']['duration'])})")

def compare_schedules(schedules_stats):
    """Compare statistics across different schedules."""
    print(f"\n{'='*100}")
    print(f"Schedule Comparison")
    print(f"{'='*100}")
    
    # Compare mean task durations
    print("\nMean Task Duration Comparison:")
    schedules_by_mean = sorted(schedules_stats.items(), 
                              key=lambda x: x[1]['task_duration_stats']['mean'])
    
    print(f"{'Schedule':<20}{'Mean Task Duration':<25}{'% of Slowest':<20}")
    print(f"{'-'*65}")
    slowest_mean = schedules_by_mean[-1][1]['task_duration_stats']['mean']
    for schedule, stats in schedules_by_mean:
        mean_duration = stats['task_duration_stats']['mean']
        percentage = (mean_duration / slowest_mean) * 100
        print(f"{schedule:<20}{format_duration(mean_duration):<25}{percentage:.2f}%")
    
    # Compare geometric means
    print("\nGeometric Mean Task Duration Comparison:")
    schedules_by_geo_mean = sorted(schedules_stats.items(), 
                                  key=lambda x: x[1]['task_duration_stats']['geo_mean'])
    
    print(f"{'Schedule':<20}{'Geo Mean Task Duration':<25}{'% of Slowest':<20}")
    print(f"{'-'*65}")
    slowest_geo_mean = schedules_by_geo_mean[-1][1]['task_duration_stats']['geo_mean']
    for schedule, stats in schedules_by_geo_mean:
        geo_mean = stats['task_duration_stats']['geo_mean']
        percentage = (geo_mean / slowest_geo_mean) * 100
        print(f"{schedule:<20}{format_duration(geo_mean):<25}{percentage:.2f}%")
    
    # Compare chunk durations across schedules
    print("\nChunk Duration Comparison:")
    chunk_count = min(stats['chunk_count_per_task'] for stats in schedules_stats.values())
    
    for chunk_id in range(chunk_count):
        print(f"\nChunk {chunk_id} Mean Duration Comparison:")
        chunk_means = [(schedule, stats['chunk_stats'][chunk_id]['mean']) 
                       for schedule, stats in schedules_stats.items()]
        chunk_means.sort(key=lambda x: x[1])
        
        print(f"{'Schedule':<20}{'Mean Duration':<25}{'% of Slowest':<20}")
        print(f"{'-'*65}")
        slowest_chunk = chunk_means[-1][1]
        for schedule, mean in chunk_means:
            percentage = (mean / slowest_chunk) * 100
            print(f"{schedule:<20}{format_duration(mean):<25}{percentage:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark files for task and chunk statistics.')
    parser.add_argument('files', metavar='FILE', nargs='*',
                        help='Benchmark files to analyze. If none provided, all non-raw files in the directory will be used.')
    parser.add_argument('--dir', default='scripts-v2/analysis',
                        help='Directory containing benchmark files (default: scripts-v2/analysis)')
    parser.add_argument('--prefix', default='BM_pipe_cifar_sparse_vk_schedule_',
                        help='File prefix to filter by (default: BM_pipe_cifar_sparse_vk_schedule_)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare statistics across different schedules')
    
    args = parser.parse_args()
    
    # Determine files to analyze
    files_to_analyze = []
    if args.files:
        files_to_analyze = args.files
    else:
        # Find all non-raw files with the given prefix
        pattern = os.path.join(args.dir, f"{args.prefix}*.txt")
        files_to_analyze = [f for f in glob.glob(pattern) if '.raw.' not in f]
    
    if not files_to_analyze:
        print(f"No benchmark files found to analyze in {args.dir}")
        return 1
    
    # Dictionary to store statistics for each schedule
    schedules_stats = {}
    
    # Process each file
    for file_path in files_to_analyze:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        tasks = parse_benchmark_file(file_path)
        if not tasks:
            print(f"Warning: No task data found in {file_name}")
            continue
        
        stats = calculate_statistics(tasks)
        if stats:
            print_statistics(file_name, stats)
            
            # Extract schedule number from filename for comparison
            match = re.search(r'schedule_(\d+)', file_name)
            if match:
                schedule_num = match.group(1)
                schedules_stats[f"Schedule {schedule_num}"] = stats
    
    # Compare schedules if requested
    if args.compare and len(schedules_stats) > 1:
        compare_schedules(schedules_stats)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 