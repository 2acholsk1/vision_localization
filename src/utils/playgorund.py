# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF

# map_path = 'data/NAIP/m_3907736_se_18_1_20170628.jpg'
# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')

# particle_number = 150
# patch_size = 127
# trajectory_length = 500
# color_bins = 16
# lbp_bins = 32

# map_cv = cv2.imread(map_path)
# map_tensor = torch.from_numpy(map_cv).float().permute(2, 0, 1) / 255.0  # [C, H, W]
# map_tensor = map_tensor.to(device)

# C, H, W = map_tensor.shape

# def generate_uav_trajectory(length, patch_size):
#     start_y = np.random.randint(patch_size//2, H - patch_size//2)
#     end_y = np.random.randint(patch_size//2, H - patch_size//2)
#     coord_heights = np.linspace(start_y, end_y, length)
#     coord_widths = np.linspace(patch_size//2, W - patch_size//2, length)
#     coords = np.stack([coord_widths, coord_heights], axis=1).astype(np.int32)
#     return coords

# def initialize_particles(num_particles, patch_size):
#     ys = torch.randint(patch_size//2, H - patch_size//2, (num_particles,), device=device)
#     xs = torch.randint(patch_size//2, W - patch_size//2, (num_particles,), device=device)
#     return torch.stack([xs, ys], dim=1)

# def get_patches(coords, map_tensor, patch_size):
#     half = patch_size // 2
#     coords = coords.clone()
#     coords[:, 0] = torch.clamp(coords[:, 0], half, map_tensor.shape[2] - half - 1)
#     coords[:, 1] = torch.clamp(coords[:, 1], half, map_tensor.shape[1] - half - 1)

#     patches = []
#     for coord in coords:
#         x, y = coord
#         patch = map_tensor[:, y-half:y+half+1, x-half:x+half+1]
#         patches.append(patch)
#     return torch.stack(patches)

# def compute_histograms(patches, color_bins=16, lbp_bins=32):
#     N, C, H, W = patches.shape
#     hists = []

#     for c in range(C):
#         channel = patches[:, c, :, :].reshape(N, -1)
#         hist = torch.stack([torch.histc(channel[i], bins=color_bins, min=0.0, max=1.0) for i in range(N)])
#         hist = hist + 1e-6
#         hist = hist / hist.sum(dim=1, keepdim=True)
#         hists.append(hist)

#     gray = 0.114 * patches[:,0,:,:] + 0.587 * patches[:,1,:,:] + 0.299 * patches[:,2,:,:]
#     gray = gray.unsqueeze(1)
#     lbps = lbp_torch(gray)
#     lbps = lbps.flatten(start_dim=2)
#     lbp_hists = torch.stack([torch.histc(lbps[i], bins=lbp_bins, min=0, max=lbp_bins-1) for i in range(N)])
#     lbp_hists = lbp_hists + 1e-6
#     lbp_hists = lbp_hists / lbp_hists.sum(dim=1, keepdim=True)

#     hists.append(lbp_hists)
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


# def systematic_resample(weights):
#     N = weights.shape[0]
#     positions = (torch.rand(1, device=device) + torch.arange(N, device=device)) / N
#     cum_sum = torch.cumsum(weights, dim=0)
#     indexes = torch.searchsorted(cum_sum, positions)
#     return indexes

# def move_particles(particles, move_model, patch_size):
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

# trajectory = generate_uav_trajectory(trajectory_length, patch_size)
# particles = initialize_particles(particle_number, patch_size)

# uav_loc = 0
# move_model = torch.zeros(2, device=device, dtype=torch.int32)

# cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Tracking", 800, 600)

# while uav_loc < trajectory_length - 1:
#     map_canvas = map_cv.copy()

#     point = trajectory[uav_loc]

#     for p in particles.cpu().numpy():
#         cv2.circle(map_canvas, (p[0], p[1]), 2, (0, 255, 255), 2)
#     cv2.circle(map_canvas, (point[0], point[1]), 6, (255, 255, 0), 3)

#     cv2.imshow('Tracking', map_canvas)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

#     template_patch = get_patches(torch.tensor([point], device=device), map_tensor, patch_size)[0]
#     particle_patches = get_patches(particles, map_tensor, patch_size)

#     scores = match_patches_batch(particle_patches, template_patch)
#     scores = scores + 1e-6
#     scores = scores / scores.sum()

#     indices = systematic_resample(scores)
#     move_model = torch.tensor(trajectory[uav_loc+1]) - torch.tensor(trajectory[uav_loc])

#     particles = move_particles(particles[indices], move_model, patch_size)

#     estimated_pos = estimate_position(particles, scores)
#     ground_truth_pos = torch.tensor(point, device=device)
#     error = compute_position_error(estimated_pos, ground_truth_pos)
#     print(f"Localization error: {error:.2f} pixels")

#     uav_loc += 1

# cv2.destroyAllWindows()
