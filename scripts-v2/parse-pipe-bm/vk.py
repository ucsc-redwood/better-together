import re
import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "vk.txt")

with open(input_file, "r") as f:
    content = f.read()

# Split content by device sections
device_pattern = r"\[\d+/\d+\]\s+Processing device:\s+([^\n]+)"
device_sections = re.split(device_pattern, content)

# First element is empty if file starts with a device section
if device_sections[0].strip() == "":
    device_sections.pop(0)

# Process each device section
results = {}
for i in range(0, len(device_sections), 2):
    if i + 1 < len(device_sections):
        device_id = device_sections[i].strip()
        device_content = device_sections[i + 1]

        # Extract benchmark data
        pattern = r"BM_run_VK_stage/(\d+)/iterations:1/repeats:5_mean\s+(\d+\.?\d*) ms"
        matches = re.findall(pattern, device_content)

        # Create dataframe for this device
        df = pd.DataFrame(matches, columns=["Stage", "Time_ms"])
        df["Stage"] = df["Stage"].astype(int)
        df["Time_ms"] = df["Time_ms"].astype(float)

        results[device_id] = df.sort_values("Stage")

# Print results for each device
for device_id, df in results.items():
    print(f"Device: {device_id}")
    print("=" * 50)
    print(df.to_string(index=False))
    print("\n")

# Check if we want to extract baseline data too
baseline_pattern = r"BM_run_VK_baseline/iterations:1/repeats:5_mean\s+(\d+\.?\d*) ms"
baseline_data = {}

for i in range(0, len(device_sections), 2):
    if i + 1 < len(device_sections):
        device_id = device_sections[i].strip()
        device_content = device_sections[i + 1]

        # Extract baseline data
        baseline_matches = re.findall(baseline_pattern, device_content)
        if baseline_matches:
            baseline_data[device_id] = float(baseline_matches[0])

if baseline_data:
    print("Baseline Comparison")
    print("=" * 50)
    for device_id, baseline_time in baseline_data.items():
        print(f"Device {device_id}: {baseline_time} ms")
