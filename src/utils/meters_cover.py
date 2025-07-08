import csv
import heapq
import os
import random
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from src.matchers.nn_matcher import NNMatcher


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

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            score = matcher.match_patches(patch)

            if len(top_matches) < top_k:
                heapq.heappush(top_matches, (score, patch, (x, y)))
            else:
                heapq.heappushpop(top_matches, (score, patch, (x, y)))

    top_matches.sort(reverse=True, key=lambda x: x[0])
    return top_matches


def calculate_covered_pixels(top_matches, patch_size, img_shape, reference_point):
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

    for _, _, (x, y) in top_matches:
        x_end = min(x + patch_size, img_shape[1])
        y_end = min(y + patch_size, img_shape[0])
        mask[y:y_end, x:x_end] = 1

    total_covered = np.sum(mask)
    ref_x, ref_y = reference_point
    inside = bool(mask[ref_y, ref_x] == 1)

    return total_covered, inside


def process_all(root_dir, output_csv="results.csv", patch_size=64, overlap=0.5, top_k=5, max_folders=100):
    matcher = NNMatcher(
        encoder_name="resnet50",
        embedding_size=128,
        weights_path="data/models/model_365.pth",
        device="cpu"
    )

    all_subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]
    selected_subdirs = random.sample(all_subdirs, min(max_folders, len(all_subdirs)))

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "image", "pixels_covered", "center_inside_top_k"])
        writer.writeheader()

        for folder in tqdm(selected_subdirs, desc=f"Processing {len(selected_subdirs)} losowych folderów"):
            try:
                images = glob(os.path.join(folder, "*.jpg")) + \
                         glob(os.path.join(folder, "*.png")) + \
                         glob(os.path.join(folder, "*.jpeg"))

                if len(images) != 1:
                    print(f"[!] Pomijam folder '{folder}', nie znaleziono dokładnie 1 obrazu.")
                    continue

                image_path = images[0]
                img = cv2.imread(image_path)
                if img is None:
                    print(f"[!] Błąd odczytu obrazu w {image_path}")
                    continue

                h, w, _ = img.shape
                center_y = h // 2
                center_x = w // 2

                uav_patch = img[
                    center_y - patch_size // 2:center_y + patch_size // 2,
                    center_x - patch_size // 2:center_x + patch_size // 2
                ]

                top_matches = find_top_matches(img, patch_size, overlap, uav_patch, matcher, top_k=top_k)
                covered, inside = calculate_covered_pixels(top_matches, patch_size, img.shape, (center_x, center_y))

                writer.writerow({
                    "folder": os.path.basename(folder),
                    "image": os.path.basename(image_path),
                    "pixels_covered": int(covered),
                    "center_inside_top_k": int(inside)
                })
                f.flush()

                status = "✅" if inside else "❌"
                print(f"[{status}] {os.path.basename(folder)}: {covered} px, środek {'jest' if inside else 'nie jest'} w top-K")

            except Exception as e:
                print(f"[BŁĄD] W folderze '{folder}': {e}")


if __name__ == "__main__":
    process_all(
        root_dir="data/test_naip",
        output_csv="results.csv",
        patch_size=64,
        overlap=0.25,
        top_k=25,
        max_folders=100
    )
