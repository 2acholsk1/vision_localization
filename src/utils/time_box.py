import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = "outputs/2025-06-23/14-58-50/results/error_steps.csv"
df = pd.read_csv(csv_path)

plt.figure(figsize=(8, 6))
sns.boxplot(y=df["step_time_s"])
plt.title("Rozkład czasu trwania jednej iteracji")
plt.ylabel("Czas iteracji [s]")
plt.grid(True)

plt.tight_layout()
plt.show()
