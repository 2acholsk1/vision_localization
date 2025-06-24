import heapq
import time

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.interpolate import CubicSpline

from src.metrics.pt_metric import MetricLogger, save_results
from src.utils.launching_utils import choose_matcher

def find_top_matches(map_picture, patch_size, overlap, uav_patch, matcher, top_k=5):
    img_h, img_w, _ = map_picture.shape
    step_size = int(patch_size * (1 - overlap))
    top_matches = []

    matcher.compute_template(uav_patch)

    for y in range(0, img_h - patch_size + step_size, step_size):
        for x in range(0, img_w - patch_size + step_size, step_size):
            y_end = min(y + patch_size, img_h)
            x_end = min(x + patch_size, img_w)
            patch = map_picture[y:y_end, x:x_end]

            score = matcher.match_patches(patch)

            if len(top_matches) < top_k:
                heapq.heappush(top_matches, (score, patch, (x, y)))
            else:
                heapq.heappushpop(top_matches, (score, patch, (x, y)))

    top_matches.sort(reverse=True, key=lambda x: x[0])
    return top_matches


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

        # LBP histogram
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


def compute_position_error(estimated_pos, ground_truth_pos):
    error = torch.norm(estimated_pos - ground_truth_pos, p=2)
    return error.item()


def estimate_position(particles, weights):
    weighted_pos = particles.float() * weights.unsqueeze(1)
    estimated_pos = weighted_pos.sum(dim=0)
    return estimated_pos


def get_patches(coords, map_tensor, patch_size):
    half = patch_size // 2
    coords = coords.clone()
    coords[:, 0] = torch.clamp(coords[:, 0], half, map_tensor.shape[2] - half - 1)
    coords[:, 1] = torch.clamp(coords[:, 1], half, map_tensor.shape[1] - half - 1)

    num = coords.shape[0]
    map_c_shape, map_height, map_width = map_tensor.shape

    offsets = torch.stack(torch.meshgrid(
        torch.arange(-half, half + 1, device=coords.device),
        torch.arange(-half, half + 1, device=coords.device),
        indexing='ij'
    ), dim=-1).reshape(-1, 2)

    all_coords = coords[:, None, :] + offsets[None, :, :]
    x = all_coords[..., 0].clamp(0, map_width - 1)
    y = all_coords[..., 1].clamp(0, map_height - 1)

    pixels = map_tensor[:, y, x]
    patches = pixels.permute(1, 0, 2).reshape(num, map_c_shape, patch_size, patch_size)
    return patches



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


def initialize_particles(num_particles, patch_size, H, W, device):
    ys = torch.randint(patch_size//2, H - patch_size//2, (num_particles,), device=device)
    xs = torch.randint(patch_size//2, W - patch_size//2, (num_particles,), device=device)
    return torch.stack([xs, ys], dim=1)


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


def match_patches_batch(descriptors, template):
    desc_hist = compute_histograms(descriptors)
    templ_hist = compute_histograms(template.unsqueeze(0))

    desc_hist = F.normalize(desc_hist, dim=1)
    templ_hist = F.normalize(templ_hist, dim=1)

    scores = torch.matmul(desc_hist, templ_hist.t()).squeeze(1)

    return scores


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


@hydra.main(config_path='configs', config_name='config_pt.yaml', version_base=None)
def main(cfg: DictConfig):
    step_count = 0

    map_path = cfg.map_path
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    particle_number = cfg.particles_num
    patch_size = cfg.patch_size
    traj_len = cfg.traj_len

    map_cv = cv2.imread(map_path)
    map_tensor = torch.from_numpy(map_cv).float().permute(2, 0, 1) / 255.0
    map_tensor = map_tensor.to(device)
    _, map_height, map_width = map_tensor.shape

    speed_mps = cfg.speed_mps
    dt_sim = cfg.dt_sim
    meters_per_pixel = 1.0
    step_size_px = speed_mps * dt_sim / meters_per_pixel


    trajectory = generate_uav_trajectory(patch_size, map_height, map_width, step_size_px, trajectory_type=cfg.trajectory_type)
    traj_len = len(trajectory)

    particles = []
    if cfg.pre_selection:
        matcher_starter = choose_matcher(
            cfg.matcher,
            cfg.encoder_name,
            cfg.embedding_size,
            cfg.weights_path
        )

        best_patches = find_top_matches(
            map_cv,
            patch_size,
            cfg.overlap,
            map_cv[trajectory[0][1] - patch_size//2 : trajectory[0][1] + patch_size//2 + 1,
                trajectory[0][0] - patch_size//2 : trajectory[0][0] + patch_size//2 + 1],
            matcher_starter,
            cfg.num_of_best_matches
        )

        particles_per_patch = particle_number // len(best_patches)

        for _, patch, (x, y) in best_patches:
            patch_h, patch_w, _ = patch.shape
            for _ in range(particles_per_patch):
                patch_x = np.random.randint(x, x + patch_w)
                patch_y = np.random.randint(y, y + patch_h)
                particles.append(torch.tensor([patch_x, patch_y], device=device))

        particles = torch.stack(particles)
    else:
        particles = initialize_particles(particle_number, patch_size, map_height, map_width, device)

    uav_loc = 0
    move_model = torch.zeros(2, device=device, dtype=torch.int32)
    metric_logger = MetricLogger()
    correct_convergences = 0
    false_convergences = 0
    total_runs = 0


    if cfg.visualize:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 800, 600)
    with torch.no_grad():
        print("STARTING")
        while uav_loc < traj_len - 1:
            real_time_start = time.time()
            step_finished = False

            while not step_finished:
                current_time = time.time()
                start_time = time.perf_counter()
                elapsed = current_time - real_time_start

                if elapsed >= cfg.dt_sim:
                    step_finished = True
                    break

                point = trajectory[uav_loc]

                template_patch = get_patches(torch.tensor([point], device=device), map_tensor, patch_size)[0]
                particle_patches = get_patches(particles, map_tensor, patch_size)

                scores = match_patches_batch(particle_patches, template_patch)
                scores = scores + 1e-6
                scores = scores / scores.sum()

                indices = systematic_resample(scores, device)
                move_model = torch.tensor(trajectory[uav_loc+1], device=device) - torch.tensor(trajectory[uav_loc], device=device)
                particles = move_particles(particles[indices], move_model, patch_size, map_height, map_width, cfg.base_noise, cfg.noise_scale_den)

                estimated_raw = estimate_position(particles, scores).float().to(device)
                step_time = time.perf_counter() - start_time

                ground_truth_pos = torch.tensor(point, device=device)
                error = compute_position_error(estimated_raw, ground_truth_pos)
                var_x = particles[:, 0].float().var().item()
                var_y = particles[:, 1].float().var().item()
                mean_var = (var_x + var_y) / 2.0
                converged = mean_var < cfg.convergence_threshold_var
                max_score = scores.max().item()
                entropy = -torch.sum(scores * torch.log(scores + 1e-8)).item()
                metric_logger.log(
                    error, ground_truth_pos, estimated_raw,
                    var_x, var_y, max_score, entropy, step_time, converged
                )

                if cfg.visualize:
                    map_canvas = map_cv.copy()
                    for i in range(1, len(trajectory)):
                        cv2.line(map_canvas, trajectory[i - 1], trajectory[i], (0, 0, 255), 10)
                    cv2.drawMarker(map_canvas, trajectory[0], (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=40, thickness=5)
                    cv2.drawMarker(map_canvas, trajectory[-1], (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=5)
                    for p in particles.cpu().numpy():
                        cv2.circle(map_canvas, (p[0], p[1]), 12, (0, 0, 0), 6)
                        cv2.circle(map_canvas, (p[0], p[1]), 8, (0, 0, 255), 4)
                    cv2.circle(map_canvas, (point[0], point[1]), 20, (255, 255, 0), 10)
                    cv2.circle(map_canvas, tuple(estimated_raw.int().tolist()), 20, (0, 255, 0), 3)

                    cv2.imshow('Tracking', map_canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

            if converged and error < cfg.error_threshold_px:
                convergence_counter += 1
                correct_convergences += 1
            elif converged and error > cfg.error_threshold_px * 2:
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
            step_count += 1
            total_runs += 1


    avg_error = sum(metric_logger.errors) / len(metric_logger.errors)
    avg_time = sum(metric_logger.step_times) / len(metric_logger.step_times)

    s = correct_convergences / total_runs if total_runs > 0 else 0.0
    f = false_convergences / total_runs if total_runs > 0 else 0.0

    print(f"METRIC_RESULTS: avg_error={avg_error:.4f}, avg_time={avg_time:.4f}, "
        f"correct_convergence={s:.4f}, false_convergence={f:.4f}")


    fig, ax = plt.subplots()
    ax.plot(metric_logger.errors)
    ax.set_title("Localization Error")
    ax.set_xlabel("Step")
    ax.set_ylabel("Error [px]")
    ax.grid(True)

    save_results(metric_logger, fig)


if __name__ == "main_pt":
    main()
