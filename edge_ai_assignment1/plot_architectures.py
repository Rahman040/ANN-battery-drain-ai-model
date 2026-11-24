import pandas as pd
import matplotlib.pyplot as plt

# Read the experiment results
df = pd.read_csv("architecture_results.csv")

# Calculate parameter counts for each architecture
def count_params(name):
    widths = [int(x) for x in name.replace("1-layer-", "").replace("2-layer-", "").split("-")]
    total = 0
    n_in = 5

    for width in widths:
        total += n_in * width + width
        n_in = width

    total += n_in * 1 + 1
    return total

df["parameters"] = df["architecture"].apply(count_params)

# Plot validation MAE against parameter count
plt.figure(figsize=(9, 6))
plt.plot(df["parameters"], df["validation_mae"], marker="o")

for _, row in df.iterrows():
    plt.annotate(
        row["architecture"],
        (row["parameters"], row["validation_mae"]),
        fontsize=8
    )

plt.xlabel("Number of parameters")
plt.ylabel("Validation MAE (days)")
plt.title("Validation MAE vs Model Parameters")
plt.grid(True)
plt.tight_layout()
plt.savefig("model/architecture_comparison.png", dpi=150)
plt.show()

print(df.to_string(index=False))
print("\nSaved model/architecture_comparison.png")



