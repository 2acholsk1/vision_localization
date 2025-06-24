import subprocess
import re
import os
import csv
from scipy.optimize import differential_evolution

LOGFILE = "de_history.csv"

if not os.path.exists(LOGFILE):
    with open(LOGFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter", "particles_num", "patch_size", "base_noise",
            "L", "avg_error", "avg_time", "correct_convergence", "false_convergence"
        ])

def run_main_with_params(params):
    run_main_with_params.iteration += 1

    particles_num, patch_size, base_noise = params
    particles_num = int(particles_num)
    patch_size = int(patch_size)
    if patch_size % 2 == 0:
        patch_size += 1
    base_noise = round(float(base_noise), 4)

    cmd = [
        "VL_PT",
        f"particles_num={particles_num}",
        f"patch_size={patch_size}",
        f"base_noise={base_noise}",
        "visualize=False"
    ]
    print(f"[ITER {run_main_with_params.iteration}] CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout

        pattern = (
            r"METRIC_RESULTS:\s*avg_error=([0-9.]+),\s*avg_time=([0-9.]+),\s*"
            r"correct_convergence=([0-9.]+),\s*false_convergence=([0-9.]+)"
        )
        match = re.search(pattern, output)

        if match:
            e = float(match.group(1))
            t = float(match.group(2))
            s = float(match.group(3))
            f = float(match.group(4))

            a, b, c, d = 1.0, 1000.0, 1000.0, 1000.0
            L = a * e + b * t - c * s + d * f

            with open(LOGFILE, "a", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([run_main_with_params.iteration, particles_num, patch_size, base_noise, L, e, t, s, f])

            print(f"[✓] L={L:.4f} | e={e:.2f}, t={t:.3f}, s={s:.2f}, f={f:.2f}")
            return L

        else:
            print("[!] Nie znaleziono metryk w outputcie.")
            print(output)
            return 9999.0

    except Exception as e:
        print(f"[!] Błąd uruchomienia: {e}")
        return 9999.0

run_main_with_params.iteration = 0

# Zakresy parametrów
bounds = [
    (50, 400),       # particles_num
    (32, 128),       # patch_size
    (0.5, 5.0)       # base_noise
]

result = differential_evolution(
    run_main_with_params,
    bounds,
    strategy='best1bin',
    maxiter=10,
    popsize=6,
    tol=0.01,
    seed=42,
    polish=True
)

print("\n=== WYNIK KOŃCOWY ===")
print("Najlepsze parametry:", result.x)
print("Najlepsza wartość funkcji celu L:", result.fun)
