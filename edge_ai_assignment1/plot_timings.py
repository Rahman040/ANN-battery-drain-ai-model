import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
df = pd.read_csv("bench_20261124_170836.csv")

# Calculate median time for each server stage
stages = ["parse", "normalise", "inference", "build"]
medians = df[stages].median()

print("Median server-stage timings:")
print(medians)

# Create timing breakdown chart
plt.figure(figsize=(8, 5))
plt.bar(medians.index, medians.values)

plt.xlabel("Server stage")
plt.ylabel("Median time (milliseconds)")
plt.title("API Server Timing Breakdown")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("model/timing_breakdown.png", dpi=150)
plt.show()

print("\nSaved model/timing_breakdown.png")
