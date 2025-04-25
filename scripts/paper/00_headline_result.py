# Measured times from the experiment

measured_times_ms = [
    4.08, 4.07, 5.10, 5.51, 7.30,
    5.76, 6.78, 7.08, 7.28, 10.69,
    9.33, 16.16, 35.56, 15.03, 13.65,
    19.59, 22.04, 26.37, 30.84, 68.88
]

# Baseline values
cpu_only_baseline = 45.8  # OMP
gpu_only_baseline = 44.9  # VK

# Compute speedups
cpu_speedups = [cpu_only_baseline / t for t in measured_times_ms]
gpu_speedups = [gpu_only_baseline / t for t in measured_times_ms]

# Display results
print("===== Speedup over Baselines =====")
print(f"{'Measured (ms)':>15}  {'CPU Speedup':>15}  {'GPU Speedup':>15}")
for t, cs, gs in zip(measured_times_ms, cpu_speedups, gpu_speedups):
    print(f"{t:15.2f}  {cs:15.2f}  {gs:15.2f}")
