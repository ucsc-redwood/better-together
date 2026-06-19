import argparse
import json

import pandas as pd
from pandasgui import show


def parse_arguments():
    parser = argparse.ArgumentParser(description="View a schedule JSON file")
    parser.add_argument("json_file", type=str, help="Path to the JSON file")
    return parser.parse_args()


def main():
    args = parse_arguments()
    json_file = args.json_file

    # Load the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    show(df)

    # Print the required fields in a clean format
    for item in data:
        uid = item.get("uid")
        metrics = item.get("metrics", {})
        max_time = metrics.get("max_time")
        min_time = metrics.get("min_time")
        avg_time = metrics.get("avg_time")
        gapness = metrics.get("gapness")

        print(f"UID: {uid}")
        print(f"  Max Time : {max_time:.2f}")
        print(f"  Min Time : {min_time:.2f}")
        print(f"  Avg Time : {avg_time:.2f}")
        print(f"  Gapness  : {gapness:.2f}")
        print("-" * 30)


if __name__ == "__main__":
    main()
