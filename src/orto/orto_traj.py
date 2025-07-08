# import csv
# from datetime import datetime
# import matplotlib.pyplot as plt
# import rasterio
# from rasterio.transform import rowcol
# from pyproj import Transformer

# def parse_date(row):
#     try:
#         return datetime.strptime(row["date_taken"], "%Y:%m:%d %H:%M:%S")
#     except:
#         return datetime.min

# def read_and_transform_points(csv_path, crs, start=0, end=None):
#     with open(csv_path, "r") as f:
#         reader = csv.DictReader(f)
#         data = [row for row in reader if row["latitude"] and row["longitude"]]
#         data.sort(key=parse_date)

#     data = data[start:end+1] if end is not None else data[start:]
#     transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

#     points = []
#     for row in data:
#         lat = float(row["latitude"])
#         lon = float(row["longitude"])
#         x, y = transformer.transform(lon, lat)
#         points.append((x, y, False))  # False = nieinterpolowany
#     return points

# def interpolate_points(points, steps=3):
#     interpolated = []
#     for i in range(len(points) - 1):
#         x0, y0, _ = points[i]
#         x1, y1, _ = points[i + 1]
#         interpolated.append((x0, y0, False))  # punkt oryginalny
#         for s in range(1, steps + 1):
#             t = s / (steps + 1)
#             x = x0 + t * (x1 - x0)
#             y = y0 + t * (y1 - y0)
#             interpolated.append((x, y, True))  # punkt interpolowany
#     interpolated.append(points[-1])  # ostatni oryginalny
#     return interpolated

# def plot_interpolated_trajectory(
#     tiff_path,
#     csv_path,
#     out_path="interpolated_trajectory.png",
#     downscale=8,
#     start_idx=0,
#     end_idx=None,
#     interpolation_steps=7
# ):
#     with rasterio.open(tiff_path) as src:
#         crs = src.crs
#         transform = src.transform

#         image = src.read(
#             out_shape=(src.count, src.height // downscale, src.width // downscale)
#         )
#         image = image.transpose(1, 2, 0)
#         image = image / image.max()

#         points = read_and_transform_points(csv_path, crs, start=start_idx, end=end_idx)
#         points = interpolate_points(points, steps=interpolation_steps)

#         plt.figure(figsize=(12, 10))
#         plt.imshow(image, origin='upper')

#         # Punkty do legendy (tylko raz)
#         drawn_original = False
#         drawn_interpolated = False

#         for x, y, is_interpolated in points:
#             row, col = rowcol(transform, x, y)
#             col //= downscale
#             row //= downscale

#             if is_interpolated:
#                 plt.plot(col, row, 'o', markersize=3, color='green',
#                          label="Punkt interpolowany" if not drawn_interpolated else "")
#                 drawn_interpolated = True
#             else:
#                 plt.plot(col, row, 'o', markersize=6, color='red',
#                          label="Punkt oryginalny" if not drawn_original else "")
#                 drawn_original = True

#         plt.title("Interpolowana trajektoria BSP")
#         plt.legend(loc='lower right', fontsize=9, frameon=True)
#         plt.axis("off")
#         plt.savefig(out_path, dpi=300, bbox_inches="tight")
#         csv_out_path = out_path.replace(".png", ".csv")
#         save_points_to_csv(points, transform, downscale, csv_out_path)

#         save_points_to_csv(points, transform, downscale, csv_out_path)

#         print(f"✅ Zapisano: {out_path}")

# import csv

# def save_points_to_csv(points, transform, downscale, output_path):
#     with open(output_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=[
#             "index", "x", "y", "col", "row", "is_interpolated"
#         ])
#         writer.writeheader()
#         for i, (x, y, is_interp) in enumerate(points):
#             row, col = rowcol(transform, x, y)
#             col //= downscale
#             row //= downscale
#             writer.writerow({
#                 "index": i,
#                 "x": x,
#                 "y": y,
#                 "col": col,
#                 "row": row,
#                 "is_interpolated": is_interp
#             })
#     print(f"✅ Zapisano punkty do CSV: {output_path}")



import matplotlib.pyplot as plt
import numpy as np
# if __name__ == "__main__":
#     plot_interpolated_trajectory(
#         tiff_path="data/20250523_EPG_4km2500m_GSD_10cm-orthomosaic.tiff",
#         csv_path="data/metadata_output_with_gps.csv",
#         out_path="data/interpolowana_trajektoria.png",
#         downscale=8,
#         start_idx=1,
#         end_idx=166,
#         interpolation_steps=70
#     )
import pandas as pd

# === ŚCIEŻKA DO PLIKU ===
csv_path = "outputs/2025-06-26/18-36-05/results/error_steps.csv"

# === WCZYTANIE ===
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Błąd wczytywania pliku: {e}")
    exit()

# === ZNAJDŹ MOMENT KONWERGENCJI ===
first_converged_index = df[df['converged'] == True].index.min()

if pd.isna(first_converged_index):
    print("Brak potwierdzonej konwergencji w danych!")
    est_df = pd.DataFrame(columns=df.columns)
else:
    est_df = df.loc[first_converged_index:]

# === ODBICIE LUSTRZANE ===
# Oblicz środek w osi Y dla odbicia
center_y = (df['gt_y'].min() + df['gt_y'].max()) / 2

# Lustrzane odbicie rzeczywistej trajektorii
mirrored_gt_y = 2 * center_y - df['gt_y']

# Lustrzane odbicie estymowanej trajektorii (po konwergencji)
if not est_df.empty:
    mirrored_est_y = 2 * center_y - est_df['est_y']

# === RYSOWANIE WYKRESU ===
plt.figure(figsize=(10, 6))

# Rzeczywista trajektoria (odbita)
plt.plot(df['gt_x'], mirrored_gt_y, label='Rzeczywista pozycja', linewidth=2)

# Estymowana trajektoria (odbita)
if not est_df.empty:
    plt.plot(est_df['est_x'], mirrored_est_y, label='Estymowana pozycja',
             linewidth=2, linestyle='--')

    # Punkty start/koniec estymowanej
    plt.scatter(est_df['est_x'].iloc[0], mirrored_est_y.iloc[0], color='orange', marker='o', label='Start (estymowana)')
    plt.scatter(est_df['est_x'].iloc[-1], mirrored_est_y.iloc[-1], color='orange', marker='x', label='Koniec (estymowana)')

# Punkty start/koniec rzeczywistej
plt.scatter(df['gt_x'].iloc[0], mirrored_gt_y.iloc[0], color='blue', marker='o', label='Start (rzeczywista)')
plt.scatter(df['gt_x'].iloc[-1], mirrored_gt_y.iloc[-1], color='blue', marker='x', label='Koniec (rzeczywista)')

plt.title('Porównanie trajektorii rzeczywistej vs. estymowanej')
plt.xlabel('X [m]')
plt.ylabel('Y [m2]')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# === STATYSTYKI PO KONWERGENCJI ===
if not est_df.empty:
    est_df['error_xy'] = np.sqrt((est_df['est_x'] - est_df['gt_x'])**2 + (est_df['est_y'] - est_df['gt_y'])**2)
    mean_error = est_df['error_xy'].mean()
    median_error = est_df['error_xy'].median()

    mean_step_time = est_df['step_time_s'].mean()
    mean_entropy = est_df['entropy'].mean()
    mean_variance = est_df['mean_var'].mean()

    print("\n=== Statystyki po konwergencji ===")
    print(f"Średni błąd estymacji pozycji: {mean_error:.2f} px (mediana: {median_error:.2f} px)")
    print(f"Średni czas przetwarzania iteracji: {mean_step_time:.4f} s")
    print(f"Średnia entropia rozkładu wag: {mean_entropy:.4f}")
    print(f"Średnia wariancja cząsteczek: {mean_variance:.2f}")
else:
    print("\nBrak danych po konwergencji – statystyki nie zostały obliczone.")
