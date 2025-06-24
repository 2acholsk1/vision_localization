import csv
import re
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from pyproj import Transformer
import numpy as np
from pathlib import Path
from PIL import Image

def extract_sort_key(filename):
    match = re.search(r'(\d{4})\D*$', filename)
    return int(match.group(1)) if match else 0

def read_sorted_trajectory(csv_path):
    points = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader if row["latitude"] and row["longitude"]]
        data.sort(key=lambda row: extract_sort_key(row["filename"]))

    for row in data:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        points.append((lat, lon))
    return points

def latlon_to_pixel_coords(lat, lon, src_crs, dst_transform):
    if src_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
    else:
        x, y = lon, lat
    col, row = rowcol(dst_transform, x, y)
    return row, col

def extract_patch(src, row, col, patch_size):
    half = patch_size // 2
    window = Window(col - half, row - half, patch_size, patch_size)
    patch = src.read(window=window)  # (C, H, W)
    patch = np.transpose(patch, (1, 2, 0))  # (H, W, C)
    return patch

def save_patch_as_image(patch, output_path):
    patch = (patch / patch.max() * 255).astype(np.uint8)
    img = Image.fromarray(patch)
    img.save(output_path)

def extract_patches_from_trajectory(
    tiff_path, csv_path, output_dir, patch_size=512
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = read_sorted_trajectory(csv_path)

    with rasterio.open(tiff_path) as src:
        for i, (lat, lon) in enumerate(points):
            row, col = latlon_to_pixel_coords(lat, lon, src.crs, src.transform)
            try:
                patch = extract_patch(src, row, col, patch_size)
                out_path = output_dir / f"patch_{i:04d}.png"
                save_patch_as_image(patch, out_path)
                print(f"Patch {i} save: {out_path.name}")
            except Exception as e:
                print(f"ERR {i}: {e}")


extract_patches_from_trajectory(
    tiff_path="data/20250523_EPG_4km2500m_GSD_10cm-orthomosaic.tiff",
    csv_path="data/EPG_photos_gps.csv",
    output_dir="data/patches_from_trajectory",
    patch_size=512
)
