#!/usr/bin/env python3

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


def visualization(map_picture, particles, particles_len, uav, metric):
    map_copy = np.copy(map_picture)
    major, minor, centroid = metric.get_shape_params(particles)

    for i in range(particles_len):
        cv2.circle(map_copy, particles[i].get_position(), 2, (0, 255, 255), 2)
        cv2.circle(map_copy, particles[i].get_position(), 4, (0, 0, 0), 2)
    cv2.ellipse(map_copy, (int(centroid[0][0]), int(centroid[0][1])),
                axes=(int(major), int(minor)), angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=2)
    cv2.circle(map_copy, (int(centroid[0][0]), int(centroid[0][1])), 5, (0, 0, 255), -1)
    cv2.circle(map_copy, uav.get_position(), 10, (255, 255, 0), 5)
    cv2.resizeWindow("Visual Localization", int(map_picture.shape[1]), int(map_picture.shape[0]))
    cv2.imshow('Visual Localization', map_copy)


@hydra.main(config_path='configs', config_name='config.yaml', version_base=None)
def main(cfg: DictConfig):
    """Main entry point of Vision Localization."""

    # cfg = OmegaConf.structured(Config())

    np.random.seed(cfg.work_env.seed)
    map_picture = cv2.imread(cfg.work_env.picture_path)
    uav = UAV(map_picture, cfg.work_env.patch_size)
    uav.generate_trajectory(cfg.uav.traj_len)

    particles = np.array([Particle(map_picture, cfg.work_env.patch_size) for _ in range(cfg.particles.number)])

    matcher = choose_matcher(cfg.matcher.name)
    resampler = choose_resampler(cfg.resampler.name, cfg.particles.number)
    euclidean_metric = EuclideanDistance()
    time_metric = TimeMetric()

    if cfg.work_env.visualize:
        cv2.namedWindow("Visual Localization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Visual Localization", int(map_picture.shape[1]), int(map_picture.shape[0]))

    while True:
        if cfg.work_env.visualize:
            visualization(map_picture, particles, cfg.particles.number, uav, euclidean_metric)
            while cfg.work_env.start < 5:
                cfg.work_env.start = cv2.waitKey(10)
            cv2.waitKey(10)
        uav.set_patch()
        matcher.compute_template_descriptor(uav.get_patch())
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

        euclidean_metric.compute(particles, uav)
        time_metric.compute()
        end_traj = uav.move()

        for particle in particles:
            particle.xy_new_swap()
            particle.move(cfg.particles.rand_static_move, uav.move_diff)

        if end_traj:
            euclidean_metric.evaluate(cfg.metrics.ploting)
            time_metric.evaluate()
            break

        save_results(euclidean_metric)


if __name__ == "__main__":
    main()
