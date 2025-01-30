#!/usr/bin/env python3

import cv2
import numpy as np

from src import config
from src.matchers.lbp_matcher import MatcherLBP
from src.metrics.euclidean_dist_metric import EuclideanDistance
from src.particle import Particle
from src.resamplers.systematic_resampler import SystematicResampler
from src.uav import UAV


def visualization(map_picture, particles, uav, metric):
    map_copy = np.copy(map_picture)
    major, minor, centroid = metric.get_shape_params(particles)

    for i in range(config.NUMBER_OF_PARTICLES):
        cv2.circle(map_copy, particles[i].get_position(), 2, (0, 255, 255), 2)
        cv2.circle(map_copy, particles[i].get_position(), 4, (0, 0, 0), 2)
    cv2.ellipse(map_copy, (int(centroid[0][0]), int(centroid[0][1])),
                axes=(int(major), int(minor)), angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=2)
    cv2.circle(map_copy, (int(centroid[0][0]), int(centroid[0][1])), 5, (0, 0, 255), -1)
    cv2.circle(map_copy, uav.get_position(), 10, (255, 255, 0), 5)
    cv2.resizeWindow("Visual Localization", int(map_picture.shape[1]), int(map_picture.shape[0]))
    cv2.imshow('Visual Localization', map_copy)


def main():
    """Main entry point of Vision Localization."""
    np.random.seed(42)
    map_picture = cv2.imread(config.MAP_PICTURE_PATH)
    uav = UAV(map_picture, config.PATCH_SIZE)
    uav.generate_trajectory(config.UAV_TRAJ_SEQ_LEN)

    particles = np.array([Particle(map_picture, config.PATCH_SIZE) for _ in range(config.NUMBER_OF_PARTICLES)])

    matcher = MatcherLBP()
    resampler = SystematicResampler(config.NUMBER_OF_PARTICLES)
    metric = EuclideanDistance()

    cv2.namedWindow("Visual Localization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visual Localization", int(map_picture.shape[1]), int(map_picture.shape[0]))

    while True:
        visualization(map_picture, particles, uav, metric)
        while config.START < 5:
            config.START = cv2.waitKey(10)
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
            particle.move(config.RAND_STATIC_MOVE, uav.move_diff)

        if end_traj:
            metric.evaluate()
            break


if __name__ == "__main__":
    main()
