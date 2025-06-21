import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.matchers.nn_matcher import NNMatcher


def save_random_patch_from_map(big_map, patch_size, output_path):
    w, h = big_map.size
    max_x = w - patch_size
    max_y = h - patch_size

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    patch = big_map.crop((x, y, x + patch_size, y + patch_size))
    patch.save(output_path)
    return patch, (x, y)

def generate_similarity_heatmap(big_map, template_patch, matcher, patch_size=63, stride=64):
    matcher.compute_template(template_patch)

    map_np = np.array(big_map)
    h, w, _ = map_np.shape

    heatmap = np.zeros(((h - patch_size) // stride + 1, (w - patch_size) // stride + 1))

    for j, y in enumerate(range(0, h - patch_size + 1, stride)):
        print(f"JOT {j}")
        for i, x in enumerate(range(0, w - patch_size + 1, stride)):
            patch = big_map.crop((x, y, x + patch_size, y + patch_size))
            similarity = matcher.match_patches(patch)
            heatmap[y // stride, x // stride] = similarity
            print(f"iii {i}")

    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap_resized = cv2.resize(heatmap_norm, (w, h), interpolation=cv2.INTER_CUBIC)

    return heatmap_resized, map_np

def overlay_heatmap_on_image(image_np, heatmap, alpha=0.5):
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def draw_patch_location_on_map(image_np, top_left, patch_size, color=(255, 0, 0), thickness=9):
    image_with_box = image_np.copy()
    x, y = top_left
    bottom_right = (x + patch_size, y + patch_size)

    image_with_box = cv2.rectangle(image_with_box, (x, y), bottom_right, color, thickness)
    return image_with_box

if __name__ == "__main__":
    matcher = NNMatcher(
        encoder_name="resnet50",
        embedding_size=128,
        weights_path="data/models/model_365.pth",
        device="cpu"
    )

    map_path = "data/NAIP/m_3907736_se_18_1_20170628.jpg"
    patch_size = 63
    stride = 48
    output_patch_path = "data/NAIP/random_template.jpg"

    big_map = Image.open(map_path).convert("RGB")
    map_np_cv = np.array(big_map)

    random_patch, patch_coords = save_random_patch_from_map(big_map, patch_size, output_patch_path)

    heatmap, map_np = generate_similarity_heatmap(big_map, random_patch, matcher, patch_size=patch_size, stride=stride)
    overlay = overlay_heatmap_on_image(map_np, heatmap)
    marked_map = draw_patch_location_on_map(map_np_cv, patch_coords, patch_size)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(marked_map)
axes[0].axis("off")
axes[0].set_xlabel("(A)", fontsize=12)

axes[1].imshow(overlay)
axes[1].axis("off")
axes[1].set_xlabel("(B)", fontsize=12)

im = axes[2].imshow(heatmap, cmap="jet")
axes[2].axis("off")
axes[2].set_xlabel("(C)", fontsize=12)
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
