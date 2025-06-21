import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def generate_uav_trajectory(patch_size, map_height, map_width, step_size_px, trajectory_type='linear', **kwargs):
    margin_y = patch_size // 2
    margin_x = patch_size // 2

    def in_bounds(x, y):
        return margin_x <= x < map_width - margin_x and margin_y <= y < map_height - margin_y

    if trajectory_type == 'linear':
        start = np.array([
            np.random.randint(margin_x, map_width - margin_x),
            np.random.randint(margin_y, map_height - margin_y)
        ])
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)

        coords = []
        current = start.copy()
        while in_bounds(current[0], current[1]):
            coords.append(current.copy())
            current = current + direction * step_size_px

    elif trajectory_type == 'sine':
        xs = np.arange(margin_x, map_width - margin_x, step_size_px)
        amplitude = (map_height - 2 * margin_y) / 4
        frequency = 2 * np.pi / len(xs)
        ys = (map_height / 2) + amplitude * np.sin(frequency * np.arange(len(xs)))
        coords = np.stack([xs, ys], axis=1)

    elif trajectory_type == 'circular':
        center_x = np.random.randint(margin_x + patch_size, map_width - margin_x - patch_size)
        center_y = np.random.randint(margin_y + patch_size, map_height - margin_y - patch_size)
        radius = kwargs.get("radius", min(map_height, map_width) // 4)

        angles = np.linspace(0, 2 * np.pi, int(2 * np.pi * radius / step_size_px))
        xs = center_x + radius * np.cos(angles)
        ys = center_y + radius * np.sin(angles)
        coords = np.stack([xs, ys], axis=1)

    elif trajectory_type == 'spline':
        start = np.array([
            np.random.randint(margin_x, map_width - margin_x),
            np.random.randint(margin_y, map_height - margin_y)
        ])
        end = np.array([
            np.random.randint(margin_x, map_width - margin_x),
            np.random.randint(margin_y, map_height - margin_y)
        ])
        mid1 = (start + end) / 2 + np.random.randn(2) * 50
        mid2 = (start + end) / 2 - np.random.randn(2) * 50

        path_points = np.stack([start, mid1, mid2, end])
        t = np.linspace(0, 1, path_points.shape[0])
        ts = np.linspace(0, 1, int(np.linalg.norm(end - start) / step_size_px))

        cs_x = CubicSpline(t, path_points[:, 0])
        cs_y = CubicSpline(t, path_points[:, 1])
        xs = cs_x(ts)
        ys = cs_y(ts)
        coords = np.stack([xs, ys], axis=1)

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    coords = np.array([pt for pt in coords if in_bounds(pt[0], pt[1])])
    coords = coords.astype(np.int32)

    return coords

map_path = "data/NAIP/m_3707807_ne_17_060_20210629.jpg"
patch_size = 64
step_size_px = 20

map_img = cv2.imread(map_path)
if map_img is None:
    raise FileNotFoundError(f"Nie znaleziono pliku: {map_path}")
map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
map_h, map_w, _ = map_img.shape

types = ['linear', 'sine', 'circular', 'spline']
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.ravel()

for i, ttype in enumerate(types):
    coords = generate_uav_trajectory(patch_size, map_h, map_w, step_size_px, trajectory_type=ttype, radius=150)
    img_copy = map_img.copy()

    for j in range(1, len(coords)):
        cv2.line(img_copy, tuple(coords[j - 1]), tuple(coords[j]), (255, 0, 0), 10)

    start_pt = tuple(coords[0])
    cv2.drawMarker(img_copy, start_pt, (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=80, thickness=9)
    cv2.putText(img_copy, "START", (start_pt[0] + 10, start_pt[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 10)

    end_pt = tuple(coords[-1])
    cv2.drawMarker(img_copy, end_pt, (255, 255, 0), markerType=cv2.MARKER_STAR, markerSize=80, thickness=9)
    cv2.putText(img_copy, "END", (end_pt[0] + 50, end_pt[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 0), 10)

    axs[i].imshow(img_copy)
    axs[i].set_title(f"Trajectory: {ttype}", fontsize=14)
    axs[i].axis('off')


plt.tight_layout()
plt.savefig("trajectories_grid.png")
plt.show()
