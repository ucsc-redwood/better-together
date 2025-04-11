from load_benchmark import get_benchmark_time, compare_benchmarks
import pandas as pd

# Load your dataframe
df = pd.read_pickle("parsed_data/3A021JEHN02756_benchmark.pkl")

# Get a specific timing
vk_stage1_time = get_benchmark_time(df, vendor="VK", stage=1)





print(f"VK Stage 1 time: {vk_stage1_time} ms")
