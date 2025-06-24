# import csv
# import heapq
# import time
# import cv2
# import hydra
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from omegaconf import DictConfig
# from pathlib import Path
# from torchvision import transforms

# from src.metrics.pt_metric import MetricLogger, save_results
# from src.utils.launching_utils import choose_matcher

# def compute_histograms(patches, color_bins=16, lbp_bins=32):
#     num, patch_c_shape, _, _ = patches.shape
#     device = patches.device
#     hists = []

#     for c in range(patch_c_shape):
#         channel = patches[:, c, :, :].contiguous().reshape(num, -1)
#         bin_edges = torch.linspace(0.0, 1.0, steps=color_bins + 1, device=device)
#         bin_indices = torch.bucketize(channel, bin_edges) - 1
#         one_hot = F.one_hot(bin_indices.clamp(0, color_bins - 1), num_classes=color_bins).float()
#         hist = one_hot.sum(dim=1) + 1e-6
#         hist = hist / hist.sum(dim=1, keepdim=True)
#         hists.append(hist)

#     gray = 0.114 * patches[:, 0, :, :] + 0.587 * patches[:, 1, :, :] + 0.299 * patches[:, 2, :, :]
#     gray = gray.unsqueeze(1)
#     lbps = lbp_torch(gray)
#     lbps = lbps.flatten(start_dim=2)

#     lbps = lbps / 255.0
#     bin_edges = torch.linspace(0, 1.0, steps=lbp_bins + 1, device=device)
#     bin_indices = torch.bucketize(lbps.contiguous(), bin_edges) - 1

#     one_hot = F.one_hot(bin_indices.clamp(0, lbp_bins - 1), num_classes=lbp_bins).float()
#     hist = one_hot.sum(dim=2).squeeze(1) + 1e-6
#     hist = hist / hist.sum(dim=1, keepdim=True)
#     hists.append(hist)

#     return torch.cat(hists, dim=1)

# def lbp_torch(x):
#     x = F.pad(x, [1, 1, 1, 1], mode='replicate')
#     center = x[:, :, 1:-1, 1:-1]
#     lbp = (x[:, :, :-2, :-2] >= center).float() * 1 + \
#           (x[:, :, :-2, 1:-1] >= center).float() * 2 + \
#           (x[:, :, :-2, 2:] >= center).float() * 4 + \
#           (x[:, :, 1:-1, 2:] >= center).float() * 8 + \
#           (x[:, :, 2:, 2:] >= center).float() * 16 + \
#           (x[:, :, 2:, 1:-1] >= center).float() * 32 + \
#           (x[:, :, 2:, :-2] >= center).float() * 64 + \
#           (x[:, :, 1:-1, :-2] >= center).float() * 128
#     return lbp

# def match_patches_batch(descriptors, template):
#     desc_hist = compute_histograms(descriptors)
#     templ_hist = compute_histograms(template.unsqueeze(0))

#     desc_hist = F.normalize(desc_hist, dim=1)
#     templ_hist = F.normalize(templ_hist, dim=1)

#     scores = torch.matmul(desc_hist, templ_hist.t()).squeeze(1)
#     return scores

# def load_reference_trajectory(csv_path, patch_dir, device, patch_size):
#     patches = []
#     coords = []

#     with open(csv_path, 'r') as f:
#         reader = csv.DictReader(f)
#         rows = sorted([r for r in reader if r['latitude'] and r['longitude']],
#                       key=lambda r: int(Path(r['filename']).stem[-4:]))

#     for i, row in enumerate(rows):
#         x = int(row['col'])
#         y = int(row['row'])
#         coords.append([x, y])

#         patch_path = Path(patch_dir) / f"patch_{i:04d}.png"
#         img = cv2.imread(str(patch_path))
#         if img is None:
#             raise FileNotFoundError(f"Brak pliku: {patch_path}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (patch_size, patch_size))
#         tensor = transforms.ToTensor()(img)
#         patches.append(tensor)

#     coords = torch.tensor(coords, dtype=torch.int32, device=device)
#     patches = torch.stack(patches).to(device)
#     return coords, patches

# def estimate_position(particles, weights):
#     weighted_pos = particles.float() * weights.unsqueeze(1)
#     estimated_pos = weighted_pos.sum(dim=0)
#     return estimated_pos

# def compute_position_error(estimated_pos, ground_truth_pos):
#     error = torch.norm(estimated_pos - ground_truth_pos, p=2)
#     return error.item()

# def initialize_particles(num_particles, patch_size, H, W, device):
#     ys = torch.randint(patch_size//2, H - patch_size//2, (num_particles,), device=device)
#     xs = torch.randint(patch_size//2, W - patch_size//2, (num_particles,), device=device)
#     return torch.stack([xs, ys], dim=1)

# def move_particles(particles, move_model, patch_size, map_height, map_width, base_noise, noise_scale_den):
#     var_x = particles[:, 0].float().var()
#     var_y = particles[:, 1].float().var()
#     mean_var = (var_x + var_y) / 2.0

#     noise_scale = torch.clamp(mean_var / noise_scale_den, 0.01, 3.0)
#     noise_amount = base_noise * noise_scale

#     noise = (torch.randn_like(particles, dtype=torch.float32) * noise_amount).round().to(torch.int32)
#     particles = particles + noise + move_model

#     particles[:, 0] = torch.clamp(particles[:, 0], patch_size//2, map_width - patch_size//2)
#     particles[:, 1] = torch.clamp(particles[:, 1], patch_size//2, map_height - patch_size//2)

#     return particles

# def systematic_resample(weights, device):
#     num = weights.shape[0]
#     positions = (torch.rand(1, device=device) + torch.arange(num, device=device)) / num
#     cum_sum = torch.cumsum(weights, dim=0)
#     indexes = torch.searchsorted(cum_sum, positions)
#     return indexes

# @hydra.main(config_path='../configs', config_name='config_orto.yaml', version_base=None)
# def main(cfg: DictConfig):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     patch_size = cfg.patch_size
#     map_path = cfg.map_path
#     particle_number = cfg.particles_num

#     map_cv = cv2.imread(map_path)
#     map_tensor = torch.from_numpy(map_cv).float().permute(2, 0, 1) / 255.0
#     map_tensor = map_tensor.to(device)
#     _, map_height, map_width = map_tensor.shape

#     trajectory, ref_patches = load_reference_trajectory(
#         csv_path=cfg.traj_csv,
#         patch_dir=cfg.traj_patch_dir,
#         device=device,
#         patch_size=patch_size
#     )
#     trajectory = trajectory.cpu()
#     traj_len = len(trajectory)

#     particles = initialize_particles(particle_number, patch_size, map_height, map_width, device)
#     move_model = torch.zeros(2, device=device, dtype=torch.int32)

#     metric_logger = MetricLogger()
#     estimated_positions = []

#     with torch.no_grad():
#         for uav_loc in range(traj_len - 1):
#             template_patch = ref_patches[uav_loc]
#             valid_patches = []
#             valid_particles = []
#             half = patch_size // 2

#             for i in range(particles.size(0)):
#                 x, y = particles[i]
#                 if (x - half >= 0 and y - half >= 0 and x + half <= map_width and y + half <= map_height):
#                     patch = map_tensor[:, y - half:y + half, x - half:x + half]
#                     valid_patches.append(patch)
#                     valid_particles.append(particles[i])

#             if len(valid_patches) == 0:
#                 print("⚠️ Brak ważnych cząstek – pomijam krok")
#                 continue

#             particle_patches = torch.stack(valid_patches)
#             particles = torch.stack(valid_particles)

#             scores = match_patches_batch(particle_patches, template_patch)
#             scores = scores + 1e-6
#             scores = scores / scores.sum()

#             indices = systematic_resample(scores, device)
#             move_model = trajectory[uav_loc + 1].float() - trajectory[uav_loc].float()
#             particles = move_particles(particles[indices], move_model.to(torch.int32), patch_size, map_height, map_width,
#                                        cfg.base_noise, cfg.noise_scale_den)

#             estimated = estimate_position(particles, scores)
#             gt = trajectory[uav_loc].float()
#             error = compute_position_error(estimated, gt)
#             estimated_positions.append(estimated.cpu().numpy())

#             metric_logger.errors.append(error)
#             print(f"Step {uav_loc} | Error: {error:.2f} px")

#     print("Avg error:", sum(metric_logger.errors) / len(metric_logger.errors))

#     # --- Wizualizacja trajektorii ---
#     map_vis = map_cv.copy()

#     for i in range(1, traj_len):
#         pt1 = tuple(trajectory[i - 1].numpy())
#         pt2 = tuple(trajectory[i].numpy())
#         cv2.line(map_vis, pt1, pt2, (255, 0, 0), 2)

#     for i in range(1, len(estimated_positions)):
#         pt1 = tuple(np.round(estimated_positions[i - 1]).astype(int))
#         pt2 = tuple(np.round(estimated_positions[i]).astype(int))
#         cv2.line(map_vis, pt1, pt2, (0, 255, 0), 2)

#     cv2.drawMarker(map_vis, tuple(trajectory[0].numpy()), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
#     cv2.drawMarker(map_vis, tuple(trajectory[-1].numpy()), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)

#     cv2.imshow("Trajectory Tracking", map_vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite("tracking_result.png", map_vis)

# if __name__ == '__main__':
#     main()

import rasterio
import cv2
import numpy as np

input_path = "data/20250523_EPG_4km2500m_GSD_10cm-orthomosaic.tiff"
output_path = "data/ortofotomapa_resized.jpg"
scale = 0.25  # np. 0.5 oznacza 50%, 0.25 = 25%

with rasterio.open(input_path) as src:
    image = src.read()  # (C, H, W)

# Zmiana formatu z (C, H, W) → (H, W, C)
image = np.transpose(image, (1, 2, 0))

# Skalowanie
resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
cv2.imwrite(output_path, resized)
