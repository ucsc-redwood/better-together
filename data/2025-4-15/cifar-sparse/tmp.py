import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base_dir, "3A021JEHN02756_normal.csv")

df_normal = pd.read_csv(filepath)

# print(df_normal)

print("--------------------------------")
print("Normal")
print("--------------------------------")

avg_table_normal = (
    df_normal.groupby("stage")[["little", "medium", "big", "vulkan"]].mean().round(4)
)
print(avg_table_normal)

filepath = os.path.join(base_dir, "3A021JEHN02756_fully.csv")
df_fully = pd.read_csv(filepath)

print("--------------------------------")
print("Fully")
print("--------------------------------")

avg_table_fully = (
    df_fully.groupby("stage")[["little", "medium", "big", "vulkan"]].mean().round(4)
)
print(avg_table_fully)

print("--------------------------------")
print("Penalty")
print("--------------------------------")

# make a pentalty table
penalty_table = avg_table_fully - avg_table_normal
print(penalty_table)

print("--------------------------------")
print("Penalty Multiplier")
print("--------------------------------")

# make a pentalty table(multplier that i can use to multiply the normal table to get the fully table)
penalty_table_multiplier = avg_table_fully / avg_table_normal
print(penalty_table_multiplier)

# Define 10 different schedules
schedules = [
    {"medium": [0], "vulkan": [1, 2], "big": [3, 4, 5, 6], "little": [7, 8]},
    {"big": [0], "vulkan": [1, 2], "medium": [3, 4, 5, 6], "little": [7, 8]},
    {"medium": [0], "vulkan": [1], "big": [2, 3, 4, 5, 6], "little": [7, 8]},
    {"vulkan": [0], "medium": [1], "big": [2, 3, 4, 5, 6], "little": [7, 8]},
    {"medium": [0], "vulkan": [1, 2], "little": [3], "big": [4, 5, 6, 7, 8]},
    {"big": [0], "vulkan": [1, 2], "little": [3], "medium": [4, 5, 6, 7, 8]},
    {"big": [0], "medium": [1], "vulkan": [2, 3, 4, 5], "little": [6, 7, 8]},
    {"big": [0], "vulkan": [1, 2], "medium": [3, 4, 5], "little": [6, 7, 8]},
    {"medium": [0], "vulkan": [1, 2], "big": [3, 4, 5], "little": [6, 7, 8]},
    {"medium": [0], "vulkan": [1], "big": [2, 3, 4, 5], "little": [6, 7, 8]},
]

# add 1 to stages
for schedule in schedules:
    for pu, stages in schedule.items():
        schedule[pu] = [stage + 1 for stage in stages]


def print_schedule_time(schedule):
    chunk_times = {}
    for pu, stages in schedule.items():
        total_time = avg_table_fully.loc[stages, pu].sum()
        chunk_times[pu] = total_time

    print("--------------------------------")
    print(f"Schedule: {schedule}")
    print("--------------------------------")
    print("Total time per chunk (using fully table):")
    for pu, time in chunk_times.items():
        print(f"\t{pu}: {time:.4f} seconds")

    print()
    print("Total time per chunk (using normal table):")
    for pu, time in chunk_times.items():
        total_time = avg_table_normal.loc[schedule[pu], pu].sum()
        print(f"\t{pu}: {total_time:.4f} seconds")


for schedule in schedules:
    print_schedule_time(schedule)
