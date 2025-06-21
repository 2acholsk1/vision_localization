import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.matchers.nn_matcher import NNMatcher


def get_random_template(big_map, patch_size, output_path):
    w, h = big_map.size
    max_x = w - patch_size
    max_y = h - patch_size

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    patch = big_map.crop((x, y, x + patch_size, y + patch_size))
    patch.save(output_path)
    print(f"Zapisano losowy patch do: {output_path}")
    return patch, (x, y)

def draw_ground_truth_location(map_np, coords, patch_size, color=(255, 0, 0), thickness=6):
    image = map_np.copy()
    x, y = coords
    cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), color, thickness)
    return image

def sliding_window_matching(map_image, matcher, template_patch, patch_size=128, stride=64, top_n=5):
    matcher.compute_template(template_patch)

    map_np = np.array(map_image)
    h, w, _ = map_np.shape

    match_scores = []
    coords = []
    i = 0
    j = 0

    for y in range(0, h - patch_size + 1, stride):
        print(f"iii {i}")
        i += 1
        j = 0
        for x in range(0, w - patch_size + 1, stride):
            patch = map_image.crop((x, y, x + patch_size, y + patch_size))
            score = matcher.match_patches(patch)
            match_scores.append(score)
            coords.append((x, y))
            print(f"jjj {j}")
            j += 1

    match_scores = np.array(match_scores)
    coords = np.array(coords)

    top_indices = np.argsort(match_scores)[-top_n:]
    top_coords = coords[top_indices]

    return top_coords, map_np

def draw_matches_on_map(map_np, coords, patch_size, color=(0, 255, 0), alpha=0.4):
    overlay = map_np.copy()

    for (x, y) in coords:
        cv2.rectangle(overlay, (x, y), (x + patch_size, y + patch_size), color, -1)

    result = cv2.addWeighted(overlay, alpha, map_np, 1 - alpha, 0)
    return result

if __name__ == "__main__":
    matcher = NNMatcher(
        encoder_name="resnet50",
        embedding_size=128,
        weights_path="data/models/model_365.pth",
        device="cpu"
    )

    map_path = "data/NAIP/m_3907736_se_18_1_20170628.jpg"
    output_patch_path = "data/template.jpg"
    output_result_path = "data/result_with_matches.jpg"
    output_truth_path = "data/ground_truth_location.jpg"

    patch_size = 256
    stride = 128
    top_n = 3

    big_map = Image.open(map_path).convert("RGB")
    template, template_coords = get_random_template(big_map, patch_size, output_patch_path)

    top_coords, map_np = sliding_window_matching(big_map, matcher, template, patch_size, stride, top_n)
    result_img = draw_matches_on_map(map_np, top_coords, patch_size, color=(0, 255, 0), alpha=0.4)

    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_result_path, result_bgr)
    print(f"{output_result_path}")

    ground_truth_img = draw_ground_truth_location(map_np, template_coords, patch_size)
    ground_bgr = cv2.cvtColor(ground_truth_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_truth_path, ground_bgr)
    print(f"{output_truth_path}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result_img)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
