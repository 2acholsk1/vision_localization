from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
import skimage.feature
from numpy.random import random

map_picture = cv.imread('data/demo/earth_pro_poznan.jpg')
# template = cv.imread('template_3.jpg')


def systematic_resample(weights):
    length = len(weights)
    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random() + np.arange(length)) / length
    indexes = np.zeros(length, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < length:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def generate_uav_trajectory(sequence_length, patch_size):
    height, width, _ = map_picture.shape
    start_point = np.random.randint(0 + int(patch_size / 2) + 1, height - int(patch_size / 2) - 1)
    end_point = np.random.randint(0 + int(patch_size / 2) + 1, height - int(patch_size / 2) - 1)
    coord_heights = np.linspace(start_point, end_point, sequence_length)
    coord_widths = np.linspace(0 + int(patch_size / 2) + 1, width - int(patch_size / 2) - 1, sequence_length)
    coordinates = []
    for i in range(sequence_length):
        coordinates.append((int(coord_widths[i]), int(coord_heights[i])))
    return np.asarray(coordinates)


def initialize_particles(particle_number, patch_size):
    height, width, _ = map_picture.shape
    particle_init_y = np.random.randint(0 + int(patch_size / 2) + 1, height - int(patch_size / 2) - 1,
                                        size=particle_number)
    particle_init_x = np.random.randint(0 + int(patch_size / 2) + 1, width - int(patch_size / 2) - 1,
                                        size=particle_number)
    particle_init = []
    for i in range(particle_number):
        particle_init.append([particle_init_x[i], particle_init_y[i]])
    return np.asarray(particle_init)


# def initialize_particles_at_loc(particle_number, loc):
#     # height, width, _ = map.shape
#     particle_init_x = np.random_randint(loc[1] - 100, loc[1] + 100, size=particle_number)
#     particle_init_y = np.random_randint(loc[0] - 100, loc[0] + 100, size=particle_number)

#     particle_init = []
#     for i in range(particle_number):
#         particle_init.append([particle_init_x[i], particle_init_y[i]])
#     return np.asarray(particle_init)


def collect_particle_desriptors(map_picture, coordinate_list, patch_size):
    descriptors = []
    for i, _ in enumerate(coordinate_list):
        x = coordinate_list[i][0]
        y = coordinate_list[i][1]
        descriptors.append(map_picture[y - int(patch_size / 2):y + int(patch_size / 2) + 1,
                           x - int(patch_size / 2):x + int(patch_size / 2) + 1])
    return descriptors


# def match_particles_against_template(particle_descriptors, template):
#     scores = []
#     for i in range(len(particle_descriptors)):
#         scores.append(cv2.matchTemplate(particle_descriptors[i], template, cv2.TM_CCOEFF_NORMED))
#         # this is where the parallel for loop comes in
#         #scores.append(match_patches(particle_descriptors[i], template))
#     scores = np.asarray(scores).reshape(len(scores), 1)
#     scores_norm = (scores + 1.0) / 2.0
#     scores_norm = scores_norm / np.sum(scores_norm)
#     return scores_norm

def match_particles_against_template(particle_descriptors, template):
    def match_single_particle(particle):
        # return cv2.matchTemplate(particle, template, cv2.TM_CCOEFF_NORMED)
        return match_patches(particle, template)

    with ThreadPoolExecutor(max_workers=40) as executor:
        # Use map to execute match_single_particle in parallel
        scores = list(executor.map(match_single_particle, particle_descriptors))

    scores = np.asarray(scores).reshape(len(scores), 1)
    scores_norm = (scores + 1.0) / 2.0
    scores_norm = scores_norm / np.sum(scores_norm)
    return scores_norm


def move_particles(indices_after_resampling, particle_coordinates, map_picture, template_size, move_model):
    new_particles = np.empty_like(particle_coordinates)
    for i, _ in enumerate(indices_after_resampling):
        new_particles[i] = particle_coordinates[indices_after_resampling[i]]
    movements = np.random.randint(-20, 20, size=new_particles.shape)
    new_particles = new_particles + movements + move_model
    new_particles[:, 0] = np.clip(new_particles[:, 0], 0 + int(template_size / 2) + 1,
                                  map_picture.shape[1] - int(template_size / 2) - 1)
    new_particles[:, 1] = np.clip(new_particles[:, 1], 0 + int(template_size / 2) + 1,
                                  map_picture.shape[0] - int(template_size / 2) - 1)
    return new_particles


def get_patch_at_coords(point, patch_size, map_picture):
    x = point[0]
    y = point[1]
    patch = map_picture[
        y - int(patch_size / 2):y + int(patch_size / 2) + 1,
        x - int(patch_size / 2):x + int(patch_size / 2) + 1
        ]
    return patch


def match_patches(candidate, template):
    # this is done in an extremely dumb way (repeatedly computing template descriptor) and needs ti be changes
    hist_candidate_b = cv.calcHist([candidate], [0], None, [16], [0, 256])
    hist_candidate_b = hist_candidate_b / hist_candidate_b.sum()
    hist_candidate_g = cv.calcHist([candidate], [1], None, [16], [0, 256])
    hist_candidate_g = hist_candidate_g / hist_candidate_g.sum()
    hist_candidate_r = cv.calcHist([candidate], [2], None, [16], [0, 256])
    hist_candidate_r = hist_candidate_r / hist_candidate_r.sum()
    hist_template_b = cv.calcHist([template], [0], None, [16], [0, 256])
    hist_template_b = hist_template_b / hist_template_b.sum()
    hist_template_g = cv.calcHist([template], [1], None, [16], [0, 256])
    hist_template_g = hist_template_g / hist_template_g.sum()
    hist_template_r = cv.calcHist([template], [2], None, [16], [0, 256])
    hist_template_r = hist_template_r / hist_template_r.sum()

    candidate_lbp = skimage.feature.local_binary_pattern(cv.cvtColor(candidate, cv.COLOR_BGR2GRAY), 8, 1,
                                                         method='nri_uniform')
    template_lbp = skimage.feature.local_binary_pattern(cv.cvtColor(template, cv.COLOR_BGR2GRAY), 8, 1,
                                                        method='nri_uniform')
    hist_candidate_lbp = cv.calcHist([candidate_lbp.astype(np.uint8)], [0], None, [32], [0, 31])
    hist_candidate_lbp = hist_candidate_lbp / hist_candidate_lbp.sum()
    hist_template_lbp = cv.calcHist([template_lbp.astype(np.uint8)], [0], None, [32], [0, 31])
    hist_template_lbp = hist_template_lbp / hist_template_lbp.sum()

    candidate_vector = np.vstack((hist_candidate_b, hist_candidate_g, hist_candidate_r, hist_candidate_lbp))
    template_vector = np.vstack((hist_template_b, hist_template_g, hist_template_r, hist_template_lbp))
    candidate_vector = candidate_vector / np.sum(candidate_vector)
    template_vector = template_vector / np.sum(template_vector)
    result = 1.0 - np.sum(abs(candidate_vector - template_vector))

    return result


# sample_1 = cv.imread('template.jpg')
# sample_2 = cv.imread('template_3.jpg')
# result_comp = match_patches(sample_1, sample_2)


# check the order of operations in the PF for correctness

UAV_LOC = 0
PARTICLE_NUMBER = 500
TEMPLATE_SIZE = 51

trajectory = generate_uav_trajectory(500, TEMPLATE_SIZE)
particle_coordinates = initialize_particles(PARTICLE_NUMBER, TEMPLATE_SIZE)
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.resizeWindow("image", int(map_picture.shape[1] / 1.3), int(map_picture.shape[0] / 1.3))

move_model = [0, 0]
KEY_CHAR = 0
while True:
    map_canvas = np.copy(map_picture)
    point = trajectory[UAV_LOC]
    for i in range(PARTICLE_NUMBER):
        cv.circle(map_canvas, particle_coordinates[i], 2, (0, 255, 255), 2)
        cv.circle(map_canvas, particle_coordinates[i], 4, (0, 0, 0), 2)
    cv.circle(map_canvas, point - move_model, 10, (255, 255, 0), 5)
    cv.resizeWindow("image", int(map_picture.shape[1] / 1.5), int(map_picture.shape[0] / 1.5))
    cv.imshow('image', map_canvas)
    while KEY_CHAR < 5:
        KEY_CHAR = cv.waitKey(10)
    cv.waitKey(10)
    UAV_LOC += 1
    template = get_patch_at_coords(point, TEMPLATE_SIZE, map_picture)
    particle_descriptors = collect_particle_desriptors(map_picture, particle_coordinates, TEMPLATE_SIZE)
    results = match_particles_against_template(particle_descriptors, template)
    indices_after_resampling = systematic_resample(results)
    move_model = trajectory[UAV_LOC] - trajectory[UAV_LOC - 1]
    print(move_model)
    particle_coordinates = move_particles(
        indices_after_resampling,
        particle_coordinates,
        map_picture,
        TEMPLATE_SIZE,
        move_model
        )
