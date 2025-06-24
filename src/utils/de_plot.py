# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# ROOT_DIR = "runs"  # ← podmień na folder z podfolderami zawierającymi de_history.csv

# def plot_run(csv_path):
#     folder = os.path.dirname(csv_path)
#     name = os.path.basename(folder)

#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         print(f"[!] Nie udało się wczytać {csv_path}: {e}")
#         return

#     if df.empty or "error_px" not in df.columns:
#         print(f"[!] Pusty lub niewłaściwy plik: {csv_path}")
#         return

#     steps = df["step"]

#     # 1. Błąd estymacji
#     plt.figure(figsize=(8, 4))
#     plt.plot(steps, df["error_px"])
#     plt.title(f"{name} – Błąd estymacji")
#     plt.xlabel("Krok")
#     plt.ylabel("Błąd [px]")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(folder, "plot_error.png"))
#     plt.close()

#     # 2. Wariancja
#     plt.figure(figsize=(8, 4))
#     plt.plot(steps, df["mean_var"], label="mean_var")
#     if "var_x" in df.columns and "var_y" in df.columns:
#         plt.plot(steps, df["var_x"], label="var_x", linestyle='--')
#         plt.plot(steps, df["var_y"], label="var_y", linestyle='--')
#     plt.title(f"{name} – Wariancja pozycji")
#     plt.xlabel("Krok")
#     plt.ylabel("Wariancja")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(folder, "plot_variance.png"))
#     plt.close()

#     # 3. Entropia i max_score
#     plt.figure(figsize=(8, 4))
#     if "entropy" in df.columns:
#         plt.plot(steps, df["entropy"], label="Entropia")
#     if "max_score" in df.columns:
#         plt.plot(steps, df["max_score"], label="Max score")
#     plt.title(f"{name} – Entropia / Max score")
#     plt.xlabel("Krok")
#     plt.ylabel("Wartość")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(folder, "plot_entropy_score.png"))
#     plt.close()

#     # 4. Zbieżność
#     if "converged" in df.columns:
#         plt.figure(figsize=(8, 3))
#         plt.plot(steps, df["converged"].astype(int), label="Zbieżność")
#         plt.title(f"{name} – Czy zbieżność?")
#         plt.xlabel("Krok")
#         plt.ylabel("1 = zbieżność")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(folder, "plot_convergence.png"))
#         plt.close()

#     print(f"[✓] Wygenerowano wykresy w: {folder}")


# def find_all_histories(root_dir):
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for name in filenames:
#             if name == "error_steps.csv":
#                 full_path = os.path.join(dirpath, name)
#                 plot_run(full_path)

# if __name__ == "__main__":
#     find_all_histories(ROOT_DIR)

# import pandas as pd
# import matplotlib.pyplot as plt

# # Wczytaj CSV
# csv_path = "de_history.csv"  # Zmień na nazwę swojej pliku
# df = pd.read_csv(csv_path)

# # Sortowanie po iteracji jeśli niepewne
# df = df.sort_values("iter")

# # Ustawienia wspólne
# plt.rcParams["figure.figsize"] = (10, 5)
# plt.rcParams["axes.grid"] = True

# # 1. Funkcja celu L
# plt.plot(df["iter"], df["L"])
# plt.xlabel("Iteracja")
# plt.ylabel("Funkcja celu L")
# plt.title("Przebieg wartości funkcji celu L")
# plt.tight_layout()
# plt.savefig("plot_L.png")
# plt.close()

# # 2. Parametry
# plt.plot(df["iter"], df["particles_num"], label="particles_num")
# plt.plot(df["iter"], df["patch_size"], label="patch_size")
# plt.plot(df["iter"], df["base_noise"], label="base_noise")
# plt.xlabel("Iteracja")
# plt.ylabel("Wartość parametru")
# plt.title("Ewolucja parametrów")
# plt.legend()
# plt.tight_layout()
# plt.savefig("plot_parameters.png")
# plt.close()

# # 3. Błąd średni
# plt.plot(df["iter"], df["avg_error"])
# plt.xlabel("Iteracja")
# plt.ylabel("Średni błąd [px]")
# plt.title("Średni błąd estymacji")
# plt.tight_layout()
# plt.savefig("plot_avg_error.png")
# plt.close()

# # 4. Zbieżności
# plt.plot(df["iter"], df["correct_convergence"], label="correct_convergence")
# plt.plot(df["iter"], df["false_convergence"], label="false_convergence")
# plt.xlabel("Iteracja")
# plt.ylabel("Udział [%]")
# plt.title("Zbieżność estymacji")
# plt.legend()
# plt.tight_layout()
# plt.savefig("plot_convergence.png")
# plt.close()

# print("✅ Wykresy zapisane jako PNG: L, parametry, avg_error, zbieżność")


import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane
csv_path = "de_history.csv"  # Zmień jeśli trzeba
df = pd.read_csv(csv_path)
df = df.sort_values("iter")

# Ustawienia
plt.rcParams["axes.grid"] = True
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 1. Funkcja celu L
axs[0].plot(df["iter"], df["L"], color="tab:blue")
axs[0].set_ylabel("Funkcja celu L")
axs[0].set_title("Funkcja celu i parametry w czasie")

# 2. Parametry
axs[1].plot(df["iter"], df["particles_num"], label="particles_num", color="tab:orange")
axs[1].plot(df["iter"], df["patch_size"], label="patch_size", color="tab:green")
axs[1].plot(df["iter"], df["base_noise"]*10, label="base_noise*10", color="tab:red")
axs[1].set_xlabel("Iteracja")
axs[1].set_ylabel("Wartości parametrów")
axs[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("plot_L_and_parameters.png")
plt.close()

print("✅ Zapisano wykres jako 'plot_L_and_parameters.png'")

