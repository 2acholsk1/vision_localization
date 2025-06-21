import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

image_path = "data/NAIP/m_3608905_nw_16_1_20140728.jpg"
patch_size = 64
num_particles = 100
overlap = 0.5
top_k = 5
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

map_img = cv2.imread(image_path)
map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
map_height, map_width, _ = map_img.shape

def generate_uav_trajectory_linear(patch_size, map_height, map_width, step_size_px, length):
    margin = patch_size // 2
    start = np.array([
        np.random.randint(margin, map_width - margin),
        np.random.randint(margin, map_height - margin)
    ])
    direction = np.random.randn(2)
    direction /= np.linalg.norm(direction)
    points = [start]
    for _ in range(length - 1):
        next_point = points[-1] + direction * step_size_px
        if (margin <= next_point[0] < map_width - margin and
            margin <= next_point[1] < map_height - margin):
            points.append(next_point)
        else:
            break
    return np.round(points).astype(int)

trajectory = generate_uav_trajectory_linear(patch_size, map_height, map_width, step_size_px=30, length=10)

def simulate_top_patches(uav_patch_center, patch_size, map_shape, top_k):
    cx, cy = uav_patch_center
    patches = []
    for _ in range(top_k):
        dx = np.random.randint(-30, 30)
        dy = np.random.randint(-30, 30)
        x = np.clip(cx + dx, patch_size//2, map_shape[1] - patch_size//2)
        y = np.clip(cy + dy, patch_size//2, map_shape[0] - patch_size//2)
        patches.append((x, y))
    return patches

uav_patch_center = trajectory[0]
top_patch_centers = simulate_top_patches(uav_patch_center, patch_size, map_img.shape, top_k=5)

particles = []
particles_per_patch = num_particles // len(top_patch_centers)
for (x, y) in top_patch_centers:
    for _ in range(particles_per_patch):
        jitter_x = np.random.randint(-patch_size//4, patch_size//4)
        jitter_y = np.random.randint(-patch_size//4, patch_size//4)
        px = np.clip(x + jitter_x, patch_size//2, map_width - patch_size//2)
        py = np.clip(y + jitter_y, patch_size//2, map_height - patch_size//2)
        particles.append((px, py))
particles = np.array(particles)

move_vector = trajectory[1] - trajectory[0]
noise = np.random.normal(0, 5, size=particles.shape)
particles_moved = particles + move_vector + noise
particles_moved = np.clip(particles_moved, patch_size//2, [map_width - patch_size//2, map_height - patch_size//2])

plt.figure(figsize=(12, 8))
plt.imshow(map_img_rgb)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajektoria UAV')
plt.scatter(particles[:, 0], particles[:, 1], color='red', label='Przed ruchem')
plt.scatter(particles_moved[:, 0], particles_moved[:, 1], color='blue', label='Po ruchu')

for i in range(len(particles)):
    plt.arrow(particles[i, 0], particles[i, 1],
              particles_moved[i, 0] - particles[i, 0],
              particles_moved[i, 1] - particles[i, 1],
              color='gray', head_width=3, head_length=4, alpha=0.5, length_includes_head=True)

plt.legend()
plt.title("Ewolucja cząsteczek z inicjalizacją przez sieć (symulowaną) i ruchem UAV")
plt.axis('off')
plt.tight_layout()
plt.show()
