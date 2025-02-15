import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_csv_files(file1, file2, file3, file4, file5, file6):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    df5 = pd.read_csv(file5)
    df6 = pd.read_csv(file6)

    plt.figure(figsize=(10, 6))
    y_values1 = np.sqrt(df1.iloc[:, 0]**2 + df1.iloc[:, 1]**2)
    y_values2 = np.sqrt(df2.iloc[:, 0]**2 + df2.iloc[:, 1]**2)

    y_values3 = np.sqrt(df3.iloc[:, 0]**2 + df3.iloc[:, 1]**2)
    y_values4 = np.sqrt(df4.iloc[:, 0]**2 + df4.iloc[:, 1]**2)

    y_values5 = np.sqrt(df5.iloc[:, 0]**2 + df5.iloc[:, 1]**2)
    y_values6 = np.sqrt(df6.iloc[:, 0]**2 + df6.iloc[:, 1]**2)
    # Tworzenie zakresu dla osi X (stepy)
    x_values = np.arange(1, 301)

    # Tworzenie wykresu
    _, ax = plt.subplots()
    plt.plot(x_values, y_values1, color='g', linestyle='-', label='Pic. 1 Error with preselection')
    plt.plot(x_values, y_values2, color='g', linestyle=':', label='Pic. 1 Error without preselection')
    plt.plot(x_values, y_values3, color='b', linestyle='-', label='Pic. 2 Error with preselection')
    plt.plot(x_values, y_values4, color='b', linestyle=':', label='Pic. 2 Error without preselection')
    plt.plot(x_values, y_values5, color='r', linestyle='-',label='Pic. 3 Error with preselection')
    plt.plot(x_values, y_values6, color='r', linestyle=':',label='Pic. 3 Error without preselection')
    plt.grid(True)
    plt.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Error [m]")
    ax.set_title("Diff between Real Position and Mean of Sum Particles localization")
    ax.set_yscale('log')
    plt.savefig('photos/pic_errors.png', dpi=300)

plot_csv_files(
    'outputs/2025-02-14/12-28-35/results/error_steps.csv',
    'outputs/2025-02-14/12-30-35/results/error_steps.csv',
    'outputs/2025-02-14/12-32-30/results/error_steps.csv',
    'outputs/2025-02-14/12-34-34/results/error_steps.csv',
    'outputs/2025-02-14/12-36-44/results/error_steps.csv',
    'outputs/2025-02-14/12-38-39/results/error_steps.csv'
    )
# 1 : 3 foto true
# 2 : 3 foto false
# 3 : 2 foto true
# 4 : 2 foto false
# 5 : 1 foto true
# 6 : 1 foto false
