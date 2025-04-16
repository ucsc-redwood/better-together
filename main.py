import re
import pandas as pd


def parse_benchmark_data(raw_data: str) -> pd.DataFrame:
    """Parse benchmark data with 4 lines per stage into a structured DataFrame."""
    pattern = re.compile(r"(\w+)=([^|]+)")
    records = []

    lines = raw_data.strip().splitlines()
    assert len(lines) % 4 == 0, "Expected 4 lines per stage."

    for i, line in enumerate(lines):
        stage = i // 4  # integer division: every 4 lines is a stage
        parsed = dict(re.findall(pattern, line))
        parsed["Stage"] = stage
        records.append(parsed)

    df = pd.DataFrame(records)

    # Convert columns to proper types
    for col in df.columns:
        if col not in ["PROCESSOR"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <benchmark_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        raw_input = f.read()

    df = parse_benchmark_data(raw_input)
    print(df.head())
