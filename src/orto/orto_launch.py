import heapq
import time

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.interpolate import CubicSpline

from src.metrics.pt_metric import MetricLogger, save_results
from src.utils.launching_utils import choose_matcher


def find_top_matches_rasterio(map_tensor, patch_size, overlap, template_patch, matcher, top_k=5):
    _, map_height, map_width = map_tensor.shape
    step_size = int(patch_size * (1 - overlap))
    top_matches = []

    matcher.compute_template(template_patch.permute(1, 2, 0).cpu().numpy())  # CxHxW → HxWxC

    for y in range(0, map_height - patch_size + 1, step_size):
        for x in range(0, map_width - patch_size + 1, step_size):
            patch = map_tensor[:, y:y+patch_size, x:x+patch_size]
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                continue
            patch_np = patch.permute(1, 2, 0).cpu().numpy()
            score = matcher.match_patches(patch_np)

            if len(top_matches) < top_k:
                heapq.heappush(top_matches, (score, (x, y)))
            else:
                heapq.heappushpop(top_matches, (score, (x, y)))

    top_matches.sort(reverse=True, key=lambda x: x[0])
    return top_matches




def load_trajectory_from_csv(csv_path, skip_interpolated=True):
    df = pd.read_csv(csv_path)
    if skip_interpolated:
        df = df[df["is_interpolated"] == False]
    trajectory = df[["col", "row"]].astype(int).values.tolist()
    return trajectory

def get_patches(coords, map_tensor, patch_size):
    half = patch_size // 2
    coords = coords.clone()
    coords[:, 0] = torch.clamp(coords[:, 0], half, map_tensor.shape[2] - half - 1)
    coords[:, 1] = torch.clamp(coords[:, 1], half, map_tensor.shape[1] - half - 1)

    num = coords.shape[0]
    map_c_shape, map_height, map_width = map_tensor.shape

    offsets = torch.stack(torch.meshgrid(
        torch.arange(-half, half, device=coords.device),
        torch.arange(-half, half, device=coords.device),
        indexing='ij'
    ), dim=-1).reshape(-1, 2)

    all_coords = coords[:, None, :] + offsets[None, :, :]
    x = all_coords[..., 0].clamp(0, map_width - 1)
    y = all_coords[..., 1].clamp(0, map_height - 1)

    pixels = map_tensor[:, y, x]
    patches = pixels.permute(1, 0, 2).reshape(num, map_c_shape, patch_size, patch_size)
    return patches

def lbp_torch(x):
    x = F.pad(x, [1,1,1,1], mode='replicate')
    center = x[:, :, 1:-1, 1:-1]
    lbp = (x[:,:, :-2, :-2] >= center).float() * 1 + \
          (x[:,:, :-2, 1:-1] >= center).float() * 2 + \
          (x[:,:, :-2, 2:  ] >= center).float() * 4 + \
          (x[:,:,1:-1, 2:  ] >= center).float() * 8 + \
          (x[:,:,2:  , 2:  ] >= center).float() * 16 + \
          (x[:,:,2:  ,1:-1] >= center).float() * 32 + \
          (x[:,:,2:  , :-2] >= center).float() * 64 + \
          (x[:,:,1:-1, :-2] >= center).float() * 128
    return lbp

def compute_histograms(patches, color_bins=16, lbp_bins=32):
    num, patch_c_shape, _, _ = patches.shape
    device = patches.device
    hists = []

    for c in range(patch_c_shape):
        channel = patches[:, c, :, :].contiguous().reshape(num, -1)
        bin_edges = torch.linspace(0.0, 1.0, steps=color_bins + 1, device=device)
        bin_indices = torch.bucketize(channel, bin_edges) - 1
        one_hot = F.one_hot(bin_indices.clamp(0, color_bins - 1), num_classes=color_bins).float()
        hist = one_hot.sum(dim=1) + 1e-6
        hist = hist / hist.sum(dim=1, keepdim=True)
        hists.append(hist)

    gray = 0.114 * patches[:, 0, :, :] + 0.587 * patches[:, 1, :, :] + 0.299 * patches[:, 2, :, :]
    gray = gray.unsqueeze(1)
    lbps = lbp_torch(gray)
    lbps = lbps.flatten(start_dim=2)

    lbps = lbps / 255.0
    bin_edges = torch.linspace(0, 1.0, steps=lbp_bins + 1, device=device)
    bin_indices = torch.bucketize(lbps.contiguous(), bin_edges) - 1

    one_hot = F.one_hot(bin_indices.clamp(0, lbp_bins - 1), num_classes=lbp_bins).float()
    hist = one_hot.sum(dim=2).squeeze(1) + 1e-6
    hist = hist / hist.sum(dim=1, keepdim=True)
    hists.append(hist)

    return torch.cat(hists, dim=1)

def match_patches_batch(descriptors, template):
    desc_hist = compute_histograms(descriptors)
    templ_hist = compute_histograms(template.unsqueeze(0))

    desc_hist = F.normalize(desc_hist, dim=1)
    templ_hist = F.normalize(templ_hist, dim=1)

    scores = torch.matmul(desc_hist, templ_hist.t()).squeeze(1)
    return scores

def estimate_position(particles, weights):
    weighted_pos = particles.float() * weights.unsqueeze(1)
    estimated_pos = weighted_pos.sum(dim=0)
    return estimated_pos

def compute_position_error(estimated_pos, ground_truth_pos):
    error = torch.norm(estimated_pos - ground_truth_pos, p=2)
    return error.item()

def initialize_particles(num_particles, patch_size, H, W, device):
    ys = torch.randint(patch_size//2, H - patch_size//2, (num_particles,), device=device)
    xs = torch.randint(patch_size//2, W - patch_size//2, (num_particles,), device=device)
    return torch.stack([xs, ys], dim=1)

def move_particles(particles, move_model, patch_size, map_height, map_width, base_noise, noise_scale_den):
    var_x = particles[:, 0].float().var()
    var_y = particles[:, 1].float().var()
    mean_var = (var_x + var_y) / 2.0

    noise_scale = torch.clamp(mean_var / noise_scale_den, 0.01, 3.0)
    noise_amount = base_noise * noise_scale

    noise = (torch.randn_like(particles, dtype=torch.float32) * noise_amount).round().to(torch.int32)
    particles = particles + noise + move_model

    particles[:, 0] = torch.clamp(particles[:, 0], patch_size//2, map_width - patch_size//2)
    particles[:, 1] = torch.clamp(particles[:, 1], patch_size//2, map_height - patch_size//2)

    return particles

def systematic_resample(weights, device):
    num = weights.shape[0]
    positions = (torch.rand(1, device=device) + torch.arange(num, device=device)) / num
    cum_sum = torch.cumsum(weights, dim=0)
    indexes = torch.searchsorted(cum_sum, positions)
    return indexes

def visualize_particles_and_trajectory_step(
    trajectory, current_particles, step_idx, ax1, ax2, ax3, map_tensor, patch_size, template_patch=None
):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    traj = np.array(trajectory)

    # Trajektoria + cząsteczki
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', label="Trajectory")
    if step_idx < len(traj):
        uav_x, uav_y = traj[step_idx]
        ax1.plot(uav_x, uav_y, 'go', markersize=8, label="UAV position")
    xs, ys = current_particles[:, 0], current_particles[:, 1]
    ax1.scatter(xs, ys, s=5, alpha=0.5, color='red', label="Particles")
    ax1.set_title(f"Step {step_idx} - Particle Filter")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    ax1.legend(loc='upper right')

    # UAV patch
    if template_patch is not None:
        patch_np = template_patch.permute(1, 2, 0).cpu().numpy()
        ax2.imshow(patch_np)
        ax2.set_title("UAV Patch")
        ax2.axis("off")

    # Losowa cząsteczka
    random_idx = np.random.randint(0, current_particles.shape[0])
    random_particle = torch.tensor(current_particles[random_idx], device=map_tensor.device).unsqueeze(0)
    random_patch = get_patches(random_particle, map_tensor, patch_size)[0]
    rand_patch_np = random_patch.permute(1, 2, 0).cpu().numpy()
    ax3.imshow(rand_patch_np)
    ax3.set_title(f"Random Particle Patch #{random_idx}")
    ax3.axis("off")

    plt.tight_layout()
    plt.pause(0.001)


@hydra.main(config_path='../configs', config_name='config_orto.yaml', version_base=None)
def main(cfg: DictConfig):
    map_path = cfg.map_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.visualize:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))


    patch_size = cfg.patch_size
    particle_number = cfg.particles_num

    # Load image using rasterio with downsampling to save memory
    with rasterio.open(map_path) as src:
        scale = cfg.downscale_factor
        indexes = [1, 2, 3]
        out_shape = (len(indexes), src.height // scale, src.width // scale)
        img_np = src.read(indexes=[1, 2, 3], out_shape=out_shape).astype(np.float32)
        img_np /= 255.0 if img_np.max() > 1.0 else 1.0
        map_tensor = torch.from_numpy(img_np).to(device)

    _, map_height, map_width = map_tensor.shape
    print(map_height)
    print(map_width)


    particles = torch.randint(0, min(map_width, map_height), (particle_number, 2), device=device)
    trajectory = load_trajectory_from_csv(cfg.trajectory_csv_path, False)
    traj_len = len(trajectory)

    if cfg.pre_selection:
        matcher = choose_matcher(
            cfg.matcher,
            cfg.encoder_name,
            cfg.embedding_size,
            cfg.weights_path
        )

        point = trajectory[0]
        template_patch = get_patches(torch.tensor([point], device=device), map_tensor, patch_size)[0]

        best_patches = find_top_matches_rasterio(
            map_tensor,
            patch_size,
            cfg.overlap,
            template_patch,
            matcher,
            cfg.num_of_best_matches
        )

        particles = []
        particles_per_patch = particle_number // len(best_patches)
        for _, (x, y) in best_patches:
            for _ in range(particles_per_patch):
                px = np.random.randint(x, x + patch_size)
                py = np.random.randint(y, y + patch_size)
                particles.append(torch.tensor([px, py], device=device))

        particles = torch.stack(particles)
    else:
        particles = initialize_particles(particle_number, patch_size, map_height, map_width, device)

    metric_logger = MetricLogger()

    uav_loc = 0
    convergence_counter = 0
    convergence_in_wrong_place = 0
    correct_convergences = 0
    false_convergences = 0
    total_runs = 0
    with torch.no_grad():
        while uav_loc < traj_len - 1:
            point = trajectory[uav_loc]

            template_patch = get_patches(torch.tensor([point], device=device), map_tensor, patch_size)[0]

            if cfg.visualize:
                visualize_particles_and_trajectory_step(
                    trajectory,
                    particles.cpu().numpy(),
                    uav_loc,
                    ax1,
                    ax2,
                    ax3,
                    map_tensor,
                    patch_size,
                    template_patch=template_patch if cfg.show_uav_patch else None
                )

            particle_patches = get_patches(particles, map_tensor, patch_size)

            scores = match_patches_batch(particle_patches, template_patch)
            scores = scores + 1e-6
            scores = scores / scores.sum()

            indices = systematic_resample(scores, device)
            move_model = torch.tensor(trajectory[uav_loc + 1], device=device) - torch.tensor(point, device=device)
            particles = move_particles(particles[indices], move_model, patch_size, map_height, map_width, cfg.base_noise, cfg.noise_scale_den)

            estimated_raw = estimate_position(particles, scores).float().to(device)
            print(f"estimated {estimated_raw}")

            ground_truth_pos = torch.tensor(point, device=device)
            print(f"GT {ground_truth_pos}")
            error = compute_position_error(estimated_raw, ground_truth_pos)
            var_x = particles[:, 0].float().var().item()
            var_y = particles[:, 1].float().var().item()
            mean_var = (var_x + var_y) / 2.0
            converged = mean_var < cfg.convergence_threshold_var
            max_score = scores.max().item()
            entropy = -torch.sum(scores * torch.log(scores + 1e-8)).item()

            metric_logger.log(
                error, ground_truth_pos, estimated_raw,
                var_x, var_y, max_score, entropy, 0, converged
            )

            if converged and error < cfg.error_threshold_px:
                convergence_counter += 1
                correct_convergences += 1
            elif converged and error > cfg.error_threshold_px * 3:
                convergence_in_wrong_place += 1
                false_convergences += 1
            else:
                convergence_counter = 0
                convergence_in_wrong_place = 0

            if convergence_counter >= cfg.required_converged_steps:
                break
            if convergence_in_wrong_place >= cfg.conv_wrong_place:
                break

            uav_loc += 1
            total_runs += 1

    avg_error = sum(metric_logger.errors) / len(metric_logger.errors)
    print(f"AVG ERROR: {avg_error:.4f}")

    save_results(metric_logger)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
