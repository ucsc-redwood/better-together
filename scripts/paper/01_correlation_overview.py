import numpy as np
import pandas as pd

# Create a dictionary to store the correlation data
correlation_data = {
    "Device": [
        "9b034f1b",
        "9b034f1b",
        "9b034f1b",
        "3A021JEHN02756",
        "3A021JEHN02756",
        "3A021JEHN02756",
    ],
    "Application": [
        "CifarDense",
        "CifarSparse",
        "Tree",
        "CifarDense",
        "CifarSparse",
        "Tree",
    ],
    "Pearson": [0.9968, 0.9684, 0.9418, 0.9990, 0.9441, 0.8450],
    "R2": [0.993, 0.9378, 0.8870, 0.9981, 0.8913, 0.7140],
}

# Create DataFrame
df = pd.DataFrame(correlation_data)

# Display the DataFrame
print(df)

# compute the geomean
geomean = np.exp(np.mean(np.log(df["Pearson"])))
print(f"Geomean: {geomean}")

# 0.9477132411663898