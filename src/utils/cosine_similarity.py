import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def lbp_torch(x):
    x = F.pad(x, [1, 1, 1, 1], mode='replicate')
    center = x[:, :, 1:-1, 1:-1]
    lbp = (x[:, :, :-2, :-2] >= center).float() * 1 + \
          (x[:, :, :-2, 1:-1] >= center).float() * 2 + \
          (x[:, :, :-2, 2:] >= center).float() * 4 + \
          (x[:, :, 1:-1, 2:] >= center).float() * 8 + \
          (x[:, :, 2:, 2:] >= center).float() * 16 + \
          (x[:, :, 2:, 1:-1] >= center).float() * 32 + \
          (x[:, :, 2:, :-2] >= center).float() * 64 + \
          (x[:, :, 1:-1, :-2] >= center).float() * 128
    return lbp

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

    gray = 0.114 * patches[:, 0, :, :] + 0.587 * patches[:, 1, :, :] + 0.299 * patches[:, 2, :, :]
    gray = gray.unsqueeze(1)
    lbps = lbp_torch(gray)
    lbps = lbps.flatten(start_dim=2)
    bin_edges = torch.linspace(0, lbp_bins, steps=lbp_bins + 1, device=device)
    bin_indices = torch.bucketize(lbps, bin_edges) - 1
    one_hot = F.one_hot(bin_indices.clamp(0, lbp_bins - 1), num_classes=lbp_bins).float()
    hist = one_hot.sum(dim=2).squeeze(1) + 1e-6
    hist = hist / hist.sum(dim=1, keepdim=True)
    hists.append(hist)

    return torch.cat(hists, dim=1)

image_path = "data/NAIP/m_3608905_nw_16_1_20140728.jpg"
num_particles = 100
patch_size = 128
device = torch.device("cpu")

img = Image.open(image_path).convert("RGB")
# img = img.resize((1024, 1024))
img_np = np.array(img)
h, w, _ = img_np.shape

patches = []
coords = []
for _ in range(num_particles):
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    patch = img_np[y:y+patch_size, x:x+patch_size]
    patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    patches.append(patch_tensor)
    coords.append((x, y))

patches_tensor = torch.cat(patches).to(device)
descs = compute_histograms(patches_tensor).cpu()

ref_idx = 0
ref_desc = descs[ref_idx].unsqueeze(0)
similarities = cosine_similarity(ref_desc.numpy(), descs.numpy())[0]
top_indices = np.argsort(similarities)[-11:][::-1]

pca = PCA(n_components=2)
descs_2d = pca.fit_transform(descs.numpy())

plt.figure(figsize=(8, 6))
plt.scatter(descs_2d[:, 0], descs_2d[:, 1], c=similarities, cmap="viridis")
plt.colorbar(label="Kosinusowe podobieństwo do cząstki referencyjnej")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()

map_copy = img_np.copy()
for idx in range(num_particles):
    x, y = coords[idx]
    cv2.circle(map_copy, (x + patch_size // 2, y + patch_size // 2), 15, (0, 0, 255), -1)

for idx in top_indices[1:]:
    x, y = coords[idx]
    cv2.rectangle(map_copy, (x, y), (x+patch_size, y+patch_size), (0, 255, 0), 4)
ref_x, ref_y = coords[ref_idx]
cv2.rectangle(map_copy, (ref_x, ref_y), (ref_x+patch_size, ref_y+patch_size), (255, 0, 0), 4)

plt.figure(figsize=(12, 8))
plt.imshow(map_copy)
plt.axis("off")
plt.show()
