import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("outputs/2025-06-23/14-58-50/results/error_steps.csv")

df_converged = df[df["converged"] == True]

plt.figure(figsize=(8, 6))
sns.boxplot(y=df_converged["error_px"])
plt.title("Rozkład błędu estymacji pozycji po osiągnięciu konwergencji")
plt.ylabel("Błąd estymacji [m]")
plt.grid(True)
plt.tight_layout()
plt.show()
