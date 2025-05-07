# import hydra
# import numpy as np
# import cv2
# import time
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# from numpy.random import random
# from omegaconf import DictConfig
# from PIL import Image



# def generate_uav_trajectory(length, patch_size, H, W):
#     start_y = np.random.randint(patch_size//2, H - patch_size//2)
#     end_y = np.random.randint(patch_size//2, H - patch_size//2)
#     coord_heights = np.linspace(start_y, end_y, length)
#     coord_widths = np.linspace(patch_size//2, W - patch_size//2, length)
#     coords = np.stack([coord_widths, coord_heights], axis=1).astype(np.int32)
#     return coords

# def initialize_particles(num_particles, patch_size, H, W, device):
#     ys = torch.randint(patch_size//2, H - patch_size//2, (num_particles,), device=device)
#     xs = torch.randint(patch_size//2, W - patch_size//2, (num_particles,), device=device)
#     return torch.stack([xs, ys], dim=1)

# def get_patches(coords, map_tensor, patch_size):
#     half = patch_size // 2
#     coords = coords.clone()
#     coords[:, 0] = torch.clamp(coords[:, 0], half, map_tensor.shape[2] - half - 1)
#     coords[:, 1] = torch.clamp(coords[:, 1], half, map_tensor.shape[1] - half - 1)

#     N = coords.shape[0]
#     C, H, W = map_tensor.shape

#     offsets = torch.stack(torch.meshgrid(
#         torch.arange(-half, half + 1, device=coords.device),
#         torch.arange(-half, half + 1, device=coords.device),
#         indexing='ij'
#     ), dim=-1).reshape(-1, 2)

#     all_coords = coords[:, None, :] + offsets[None, :, :]
#     x = all_coords[..., 0].clamp(0, W - 1)
#     y = all_coords[..., 1].clamp(0, H - 1)

#     pixels = map_tensor[:, y, x]
#     patches = pixels.permute(1, 0, 2).reshape(N, C, patch_size, patch_size)
#     return patches

# def compute_histograms(patches, color_bins=16, lbp_bins=32):
#     N, C, H, W = patches.shape
#     device = patches.device
#     hists = []

#     for c in range(C):
#         channel = patches[:, c, :, :].reshape(N, -1)
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
#     bin_edges = torch.linspace(0, lbp_bins, steps=lbp_bins + 1, device=device)
#     bin_indices = torch.bucketize(lbps, bin_edges) - 1
#     one_hot = F.one_hot(bin_indices.clamp(0, lbp_bins - 1), num_classes=lbp_bins).float()
#     hist = one_hot.sum(dim=2).squeeze(1) + 1e-6
#     hist = hist / hist.sum(dim=1, keepdim=True)
#     hists.append(hist)

#     return torch.cat(hists, dim=1)

# def lbp_torch(x):
#     x = F.pad(x, [1,1,1,1], mode='replicate')
#     center = x[:, :, 1:-1, 1:-1]
#     lbp = (x[:,:, :-2, :-2] >= center).float() * 1 + \
#           (x[:,:, :-2, 1:-1] >= center).float() * 2 + \
#           (x[:,:, :-2, 2:  ] >= center).float() * 4 + \
#           (x[:,:,1:-1, 2:  ] >= center).float() * 8 + \
#           (x[:,:,2:  , 2:  ] >= center).float() * 16 + \
#           (x[:,:,2:  ,1:-1] >= center).float() * 32 + \
#           (x[:,:,2:  , :-2] >= center).float() * 64 + \
#           (x[:,:,1:-1, :-2] >= center).float() * 128
#     return lbp

# def match_patches_batch(descriptors, template):
#     desc_hist = compute_histograms(descriptors)
#     templ_hist = compute_histograms(template.unsqueeze(0))

#     desc_hist = F.normalize(desc_hist, dim=1)
#     templ_hist = F.normalize(templ_hist, dim=1)

#     scores = torch.matmul(desc_hist, templ_hist.t()).squeeze(1)

#     return scores


# def systematic_resample(weights, device):
#     N = weights.shape[0]
#     positions = (torch.rand(1, device=device) + torch.arange(N, device=device)) / N
#     cum_sum = torch.cumsum(weights, dim=0)
#     indexes = torch.searchsorted(cum_sum, positions)
#     return indexes

# def move_particles(particles, move_model, patch_size, H, W):
#     var_x = particles[:, 0].float().var()
#     var_y = particles[:, 1].float().var()
#     mean_var = (var_x + var_y) / 2.0

#     base_noise = 5.0
#     noise_scale = torch.clamp(mean_var / 5000.0, 0.5, 3.0)
#     noise_amount = base_noise * noise_scale

#     noise = (torch.randn_like(particles, dtype=torch.float32) * noise_amount).round().to(torch.int32)
#     particles = particles + noise + move_model

#     particles[:, 0] = torch.clamp(particles[:, 0], patch_size//2, W - patch_size//2)
#     particles[:, 1] = torch.clamp(particles[:, 1], patch_size//2, H - patch_size//2)
#     return particles

# def estimate_position(particles, weights):
#     weighted_pos = particles.float() * weights.unsqueeze(1)
#     estimated_pos = weighted_pos.sum(dim=0)
#     return estimated_pos

# def compute_position_error(estimated_pos, ground_truth_pos):
#     error = torch.norm(estimated_pos - ground_truth_pos, p=2)
#     return error.item()

# @hydra.main(config_path='configs', config_name='config_pt.yaml', version_base=None)
# def main(cfg: DictConfig):
#     total_time = 0.0
#     step_count = 0

#     map_path = 'data/NAIP/m_3907736_se_18_1_20170628.jpg'
#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if use_cuda else 'cpu')

#     particle_number = 150
#     patch_size = 63
#     trajectory_length = 500
#     color_bins = 16
#     lbp_bins = 32

#     map_cv = cv2.imread(map_path)
#     map_tensor = torch.from_numpy(map_cv).float().permute(2, 0, 1) / 255.0  # [C, H, W]
#     map_tensor = map_tensor.to(device)

#     C, H, W = map_tensor.shape
#     trajectory = generate_uav_trajectory(trajectory_length, patch_size, H, W)
#     particles = initialize_particles(particle_number, patch_size, H, W, device)
#     uav_loc = 0
#     move_model = torch.zeros(2, device=device, dtype=torch.int32)
#     errors = []

#     cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Tracking", 800, 600)
#     with torch.no_grad():
#         while uav_loc < trajectory_length - 1:
#             torch.cuda.synchronize(); t0 = time.time()

#             map_canvas = map_cv.copy()
#             point = trajectory[uav_loc]

#             for p in particles.cpu().numpy():
#                 cv2.circle(map_canvas, (p[0], p[1]), 2, (0, 255, 255), 2)
#             cv2.circle(map_canvas, (point[0], point[1]), 6, (255, 255, 0), 3)

#             cv2.imshow('Tracking', map_canvas)
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#             torch.cuda.synchronize(); t1 = time.time()
#             template_patch = get_patches(torch.tensor([point], device=device), map_tensor, patch_size)[0]
#             particle_patches = get_patches(particles, map_tensor, patch_size)
#             torch.cuda.synchronize(); t2 = time.time()

#             scores = match_patches_batch(particle_patches, template_patch)
#             scores = scores + 1e-6
#             scores = scores / scores.sum()
#             torch.cuda.synchronize(); t3 = time.time()

#             indices = systematic_resample(scores, device)

#             move_model = torch.tensor(trajectory[uav_loc+1], device=device)
#  - torch.tensor(trajectory[uav_loc], device=device)
#             particles = move_particles(particles[indices], move_model, patch_size, H, W)
#             torch.cuda.synchronize(); t4 = time.time()

#             estimated_pos = estimate_position(particles, scores)
#             ground_truth_pos = torch.tensor(point, device=device)
#             error = compute_position_error(estimated_pos, ground_truth_pos)
#             errors.append(error)
#             torch.cuda.synchronize(); t5 = time.time()

#             total_step = t5 - t0
#             patch_time = t2 - t1
#             match_time = t3 - t2
#             move_time = t4 - t3
#             error_time = t5 - t4

#             print(f"[Step {uav_loc:03d}] patch: {patch_time*1000:.1f} ms | match: {match_time*1000:.1f} ms |
# move: {move_time*1000:.1f} ms | error: {error_time*1000:.1f} ms | 
# total: {total_step*1000:.1f} ms | err: {error:.2f} px")

#             total_time += total_step
#             step_count += 1
#             uav_loc += 1

#     avg_step_time = total_time / step_count if step_count > 0 else 0
#     avg_error = sum(errors) / len(errors)
#     print(f"\nAverage step time: {avg_step_time*1000:.2f} ms")
#     print(f"Average localization error: {avg_error:.2f} px")



# if __name__ == "main_pt":
#     main()
