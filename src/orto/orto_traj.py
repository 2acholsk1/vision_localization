import csv
import re
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import matplotlib.pyplot as plt
from pathlib import Path

def extract_sort_key(filename):
    match = re.search(r'(\d{4})\D*$', filename)
    return int(match.group(1)) if match else 0

def read_sorted_points(csv_path, crs, start=0, end=None):
    points = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader if row["latitude"] and row["longitude"]]
        data.sort(key=lambda row: extract_sort_key(row["filename"]))

    data = data[start:end+1] if end is not None else data[start:]

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    for row in data:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        x, y = transformer.transform(lon, lat)
        points.append((x, y))
    return points

def plot_ortho_with_partial_trajectory(
    tiff_path, csv_path, out_path=None, downscale=8, start_idx=26, end_idx=162
):
    with rasterio.open(tiff_path) as src:
        crs = src.crs
        transform = src.transform

        # Zczytaj zeskalowaną mapę (mniej RAM-u)
        data = src.read(
            out_shape=(src.count, src.height // downscale, src.width // downscale)
        )
        data = data.transpose(1, 2, 0)
        data = data / data.max()

        points = read_sorted_points(csv_path, crs, start=start_idx, end=end_idx)

        pixel_points = []
        for x, y in points:
            col, row = rowcol(transform, x, y)
            col //= downscale
            row //= downscale
            pixel_points.append((col, row))

        # Rysuj
        plt.figure(figsize=(12, 10))
        plt.imshow(data)
        for i, (x, y) in enumerate(pixel_points):
            plt.plot(x, y, 'ro', markersize=4)
            plt.text(x + 4, y - 4, str(start_idx + i), fontsize=7, color='white')
        plt.title(f"Trajektoria UAV (punkty {start_idx}–{end_idx})")
        plt.axis("off")

        if out_path:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"📷 Zapisano: {out_path}")
        else:
            plt.show()

# 🔧 Przykład użycia:
plot_ortho_with_partial_trajectory(
    tiff_path="data/20250523_EPG_4km2500m_GSD_10cm-orthomosaic.tiff",
    csv_path="data/EPG_photos_gps.csv",
    out_path="data/ortho_trajektoria_26_162.png",
    downscale=8,
    start_idx=26,
    end_idx=162
)
