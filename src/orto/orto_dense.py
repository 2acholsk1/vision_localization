import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window


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

def densify_trajectory_with_originals(points, num_between=3):
    full = []
    original_indices = []
    idx_counter = 0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        xs = np.linspace(x0, x1, num_between + 2)
        ys = np.linspace(y0, y1, num_between + 2)
        segment = list(zip(xs, ys))
        for j, pt in enumerate(segment[:-1]):
            full.append(pt)
            if j == 0:
                original_indices.append(idx_counter)
            idx_counter += 1
    full.append(points[-1])
    original_indices.append(len(full) - 1)
    return full, original_indices

def extract_patches_from_tiff(tiff_path, points, out_dir, patch_size=64):
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(tiff_path) as src:
        transform = src.transform
        count = 0
        for idx, (x, y) in enumerate(points):
            col, row = rowcol(transform, x, y)

            half = patch_size // 2
            row_start = max(row - half, 0)
            row_end = min(row + half, src.height)
            col_start = max(col - half, 0)
            col_end = min(col + half, src.width)

            window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
            patch = src.read(indexes=(1, 2, 3), window=window)

            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                continue  # za mały przy brzegach

            patch = np.transpose(patch, (1, 2, 0))
            patch = (patch / patch.max() * 255).astype(np.uint8)
            image = Image.fromarray(patch)
            image.save(os.path.join(out_dir, f"patch_{count:04d}.png"))
            count += 1
    print(f"✅ Zapisano {count} patchy do: {out_dir}")

def save_patch_metadata(points, original_indices, out_csv_path):
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "type"])
        for idx, (x, y) in enumerate(points):
            point_type = "original" if idx in original_indices else "interpolated"
            writer.writerow([idx, x, y, point_type])
    print(f"📄 Zapisano metadane do: {out_csv_path}")

def plot_trajectory_with_ortho(tiff_path, dense_points, original_indices, buffer=200, downscale=4):
    with rasterio.open(tiff_path) as src:
        transform = src.transform

        xs, ys = zip(*dense_points)
        minx, maxx = min(xs) - buffer, max(xs) + buffer
        miny, maxy = min(ys) - buffer, max(ys) + buffer

        col_min, row_min = rowcol(transform, minx, maxy)
        col_max, row_max = rowcol(transform, maxx, miny)

        row_min = max(0, row_min)
        row_max = min(src.height, row_max)
        col_min = max(0, col_min)
        col_max = min(src.width, col_max)

        height = row_max - row_min
        width = col_max - col_min

        window = Window(col_min, row_min, width, height)
        out_shape = (3, height // downscale, width // downscale)

        data = src.read(indexes=(1, 2, 3), window=window, out_shape=out_shape, resampling=Resampling.bilinear)
        ortho_patch = np.transpose(data, (1, 2, 0)).astype(np.float32)
        ortho_patch /= ortho_patch.max()

        top_left = transform * (col_min, row_min)
        bottom_right = transform * (col_max, row_max)

        extent = (top_left[0], bottom_right[0], bottom_right[1], top_left[1])

        plt.figure(figsize=(12, 10))
        plt.imshow(ortho_patch, extent=extent, origin='upper')

        x_all, y_all = zip(*dense_points)
        x_orig = [x_all[i] for i in original_indices]
        y_orig = [y_all[i] for i in original_indices]

        plt.plot(x_all, y_all, '-', color='white', linewidth=1, label="trajektoria")
        plt.scatter(x_all, y_all, color='yellow', s=12, label="interpolowane")
        plt.scatter(x_orig, y_orig, color='red', s=20, label="oryginalne")
        plt.title("Trajektoria UAV na ortofotomapie")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

# === PARAMETRY ===
tiff_path = "data/20250523_EPG_4km2500m_GSD_10cm-orthomosaic.tiff"
csv_path = "data/EPG_photos_gps.csv"
out_dir = "data/patches_dense"
metadata_csv_path = os.path.join(out_dir, "metadata.csv")
start_idx = 30
end_idx = 166
patch_size = 64
num_between = 3

# === WYKONANIE ===
with rasterio.open(tiff_path) as src:
    crs = src.crs

points = read_sorted_points(csv_path, crs, start=start_idx, end=end_idx)
dense_points, original_indices = densify_trajectory_with_originals(points, num_between=num_between)
extract_patches_from_tiff(tiff_path, dense_points, out_dir, patch_size=patch_size)
save_patch_metadata(dense_points, original_indices, metadata_csv_path)
plot_trajectory_with_ortho(tiff_path, dense_points, original_indices, buffer=200, downscale=4)
