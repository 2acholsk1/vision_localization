#!/usr/bin/env python3

import os
import shutil

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf

from src.configs.config import Config
from src.matchers.lbp_matcher import MatcherLBP
from src.metrics.euclidean_dist_metric import EuclideanDistance
from src.particle import Particle
from src.resamplers.systematic_resampler import SystematicResampler
from src.uav import UAV


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


@hydra.main(config_name=None, version_base=None)
def main(_):
    """Main entry point of Vision Localization."""

    cfg = OmegaConf.structured(Config())

    np.random.seed(cfg.work_env.seed)
    map_picture = cv2.imread(cfg.work_env.picture_path)
    uav = UAV(map_picture, cfg.work_env.patch_size)
    uav.generate_trajectory(cfg.uav.traj_len)

    particles = np.array([Particle(map_picture, cfg.work_env.patch_size) for _ in range(cfg.particles.number)])

    matcher = MatcherLBP()
    resampler = SystematicResampler(cfg.particles.number)
    metric = EuclideanDistance()

    cv2.namedWindow("Visual Localization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visual Localization", int(map_picture.shape[1]), int(map_picture.shape[0]))

    while True:
        visualization(map_picture, particles, cfg.particles.number, uav, metric)
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

        metric.compute(particles, uav)
        end_traj = uav.move()

        for particle in particles:
            particle.xy_new_swap()
            particle.move(cfg.particles.rand_static_move, uav.move_diff)

        if end_traj:
            metric.evaluate()
            break

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config_file_path = os.path.abspath("src/configs/config.py")

    shutil.copy(config_file_path, os.path.join(output_dir, "config.py"))


if __name__ == "__main__":
    main()
