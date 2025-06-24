import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from pyproj import Transformer
import numpy as np

# 📍 Parametry
patch_size = 2048
x, y = 8000, 13000  # kolumna, wiersz
tiff_path = "data/20250523_EPG_4km2500m_GSD_10cm-orthomosaic.tiff"

with rasterio.open(tiff_path) as src:
    window = Window(x, y, patch_size, patch_size)
    patch = src.read(window=window)  # shape: (C, H, W)

    # Środek patcha w pikselach
    center_row = y + patch_size // 2
    center_col = x + patch_size // 2

    # Współrzędne z transformacji (mogą być metryczne lub geograficzne)
    lon, lat = xy(src.transform, center_row, center_col)

    # Jeśli CRS to nie WGS84 (EPSG:4326), przelicz
    if src.crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(lon, lat)

    print("📍 Pozycja środka patcha (lat, lon):")
    print(f"{lat:.6f}, {lon:.6f}")

# 🖼️ Wyświetlanie patcha
patch = np.transpose(patch, (1, 2, 0))  # (H, W, C)
patch = patch / 255.0 if patch.max() > 1.0 else patch

plt.figure(figsize=(10, 10))
plt.imshow(patch)
plt.title("Patch z ortofotomapy")
plt.axis("off")
plt.show()
