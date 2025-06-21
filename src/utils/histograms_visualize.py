import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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

    lbps = lbps / 255.0
    bin_edges = torch.linspace(0, 1.0, steps=lbp_bins + 1, device=device)
    bin_indices = torch.bucketize(lbps.contiguous(), bin_edges) - 1

    one_hot = F.one_hot(bin_indices.clamp(0, lbp_bins - 1), num_classes=lbp_bins).float()
    hist = one_hot.sum(dim=2).squeeze(1) + 1e-6
    hist = hist / hist.sum(dim=1, keepdim=True)
    hists.append(hist)

    return torch.cat(hists, dim=1)

def visualize_patch_histograms(image_path, patch_size=128):
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    patch = img_np[y:y+patch_size, x:x+patch_size]

    patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    hist = compute_histograms(patch_tensor)
    color_bins = 16
    lbp_bins = 32

    r_hist = hist[0, 0:color_bins].cpu().numpy()
    g_hist = hist[0, color_bins:2*color_bins].cpu().numpy()
    b_hist = hist[0, 2*color_bins:3*color_bins].cpu().numpy()
    lbp_hist = hist[0, 3*color_bins:].cpu().numpy()

    fig = plt.figure(figsize=(14, 8))

    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax1.imshow(patch)
    ax1.set_title("Wycięty fragment obrazu")
    ax1.axis("off")

    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax2.bar(np.arange(lbp_bins), lbp_hist, color='black')
    ax2.set_title("Histogram LBP")

    ax3 = plt.subplot2grid((2, 3), (1, 0))
    ax3.bar(np.arange(color_bins), r_hist, color='red')
    ax3.set_title("Histogram R")

    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax4.bar(np.arange(color_bins), g_hist, color='green')
    ax4.set_title("Histogram G")

    ax5 = plt.subplot2grid((2, 3), (1, 2))
    ax5.bar(np.arange(color_bins), b_hist, color='blue')
    ax5.set_title("Histogram B")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "data/NAIP/m_3608905_nw_16_1_20140728.jpg"
    visualize_patch_histograms(image_path, patch_size=128)
