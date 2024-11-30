#!/usr/bin/env python3

import cv2
import numpy as np

from src import config
from src.matchers.lbp_matcher import MatcherLBP
from src.particle import Particle
from src.resamplers.systematic_resampler import SystematicResampler
from src.uav import UAV


def visualization(map_picture, particles, uav):
    map_copy = np.copy(map_picture)
    for i in range(config.NUMBER_OF_PARTICLES):
        cv2.circle(map_copy, particles[i].get_position(), 2, (0, 255, 255), 2)
        cv2.circle(map_copy, particles[i].get_position(), 4, (0, 0, 0), 2)
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

    cv2.namedWindow("Visual Localization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visual Localization", int(map_picture.shape[1]), int(map_picture.shape[0]))

    while True:
        visualization(map_picture, particles, uav)
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

        new_indices = resampler.resampling(particles)
        for i, particle in enumerate(particles):
            particles[i].x_new = particles[new_indices[i]].x
            particles[i].y_new = particles[new_indices[i]].y

        end_traj = uav.move()

        for particle in particles:
            particle.xy_new_swap()
            particle.move(config.RAND_STATIC_MOVE, uav.move_diff)

        if end_traj:
            break


if __name__ == "__main__":
    main()
