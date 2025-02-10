#!/usr/bin/env python3

import heapq
import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from src.metrics.euclidean_dist_metric import EuclideanDistance
from src.metrics.time_metric import TimeMetric
from src.particle import Particle
from src.uav import UAV
from src.utils.hydra_saving import save_results
from src.utils.launching_utils import choose_matcher, choose_resampler


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


def visualize_particles(map_picture, particles, uav, metric):
    map_copy = np.copy(map_picture)
    major, minor, centroid = metric.get_shape_params(particles)

    for particle in particles:
        cv2.circle(map_copy, particle.get_position(), 2, (0, 255, 255), 2)
        cv2.circle(map_copy, particle.get_position(), 4, (0, 0, 0), 2)

    cv2.ellipse(
        map_copy, (int(centroid[0][0]), int(centroid[0][1])),
        axes=(int(major), int(minor)), angle=0, startAngle=0, endAngle=360,
        color=(0, 255, 0), thickness=2
    )
    cv2.circle(map_copy, (int(centroid[0][0]), int(centroid[0][1])), 5, (0, 0, 255), -1)
    cv2.circle(map_copy, uav.get_position(), 10, (255, 255, 0), 5)

    cv2.imshow('Visual Localization', map_copy)
    cv2.waitKey(1)


def compute_matching(particles, uav, matcher, resampler):
    uav.set_patch()
    matcher.compute_template(uav.get_patch())

    for particle in particles:
        particle.set_patch()
        particle.weight = matcher.match_patches(particle.get_patch())

    sum_of_weights = matcher.get_sum_of_weight()

    for particle in particles:
        particle.weight = particle.weight / sum_of_weights
        particle.last_weight = particle.weight

    new_indices = resampler.resampling(particles)
    for i, particle in enumerate(particles):
        particles[i].x_new = particles[new_indices[i]].x
        particles[i].y_new = particles[new_indices[i]].y


@hydra.main(config_path='configs', config_name='config.yaml', version_base=None)
def main(cfg: DictConfig):
    """Main entry point of Vision Localization."""
    np.random.seed(cfg.work_env.seed)
    map_picture = cv2.imread(cfg.work_env.picture_path)

    euclidean_metric = EuclideanDistance()
    time_metric = TimeMetric()

    uav = UAV(map_picture, cfg.work_env.patch_size)
    uav.generate_trajectory(cfg.uav.traj_len)

    matcher_starter = choose_matcher(
        cfg.matcher.starter,
        cfg.model.encoder_name,
        cfg.model.embedding_size,
        cfg.model.weights_path
    )
    matcher_cont = choose_matcher(
        cfg.matcher.continous,
        cfg.model.encoder_name,
        cfg.model.embedding_size,
        cfg.model.weights_path
    )
    resampler = choose_resampler(cfg.resampler.name, cfg.particles.number)

    uav.set_patch()
    best_patches = find_top_matches(
        map_picture,
        cfg.matcher.patch_size,
        cfg.matcher.overlap,
        uav.get_patch(),
        matcher_starter,
        cfg.matcher.num_of_best_matches
    )

    particles = []
    particles_per_patch = cfg.particles.number // len(best_patches)
    for _, patch, (x, y) in best_patches:
        patch_h, patch_w, _ = patch.shape
        for _ in range(particles_per_patch):
            patch_x = np.random.randint(x, x + patch_w)
            patch_y = np.random.randint(y, y + patch_h)
            particles.append(Particle(map_picture, cfg.work_env.patch_size, position=(patch_x, patch_y)))

    particles = np.array(particles)

    if cfg.work_env.visualize:
        cv2.namedWindow("Visual Localization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Visual Localization", int(map_picture.shape[1]), int(map_picture.shape[0]))

    while True:
        if cfg.work_env.visualize:
            visualize_particles(map_picture, particles, uav, euclidean_metric)

        compute_matching(particles, uav, matcher_cont, resampler)

        euclidean_metric.compute(particles, uav)
        time_metric.compute()

        end_traj = uav.move()

        for particle in particles:
            particle.xy_new_swap()
            particle.move(cfg.particles.rand_static_move, uav.move_diff)

        if end_traj:
            fig = euclidean_metric.evaluate(cfg.metrics.ploting)
            time_metric.evaluate()
            save_results(euclidean_metric, fig)
            break



if __name__ == "__main__":
    main()
