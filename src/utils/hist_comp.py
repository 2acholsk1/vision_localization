import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import local_binary_pattern


def lbp_torch(x):
    # pad image for 3x3 mask size
    x = F.pad(input=x, pad=[1, 1, 1, 1], mode='constant')
    b = x.shape
    M = b[1]  # pylint: disable=invalid-name
    N = b[2]  # pylint: disable=invalid-name

    y = x
    # select elements within 3x3 mask
    y00 = y[:, 0:M-2, 0:N-2]
    y01 = y[:, 0:M-2, 1:N-1]
    y02 = y[:, 0:M-2, 2:N]

    y10 = y[:, 1:M-1, 0:N-2]
    y11 = y[:, 1:M-1, 1:N-1]
    y12 = y[:, 1:M-1, 2:N]

    y20 = y[:, 2:M, 0:N-2]
    y21 = y[:, 2:M, 1:N-1]
    y22 = y[:, 2:M, 2:N]

    # Apply comparisons and multiplications
    bit = torch.ge(y00, y11)
    tmp = torch.mul(bit, torch.tensor(1))

    bit = torch.ge(y01, y11)
    val = torch.mul(bit, torch.tensor(2))
    val = torch.add(val, tmp)

    bit = torch.ge(y02, y11)
    tmp = torch.mul(bit, torch.tensor(4))
    val = torch.add(val, tmp)

    bit = torch.ge(y12, y11)
    tmp = torch.mul(bit, torch.tensor(8))
    val = torch.add(val, tmp)

    bit = torch.ge(y22, y11)
    tmp = torch.mul(bit, torch.tensor(16))
    val = torch.add(val, tmp)

    bit = torch.ge(y21, y11)
    tmp = torch.mul(bit, torch.tensor(32))
    val = torch.add(val, tmp)

    bit = torch.ge(y20, y11)
    tmp = torch.mul(bit, torch.tensor(64))
    val = torch.add(val, tmp)

    bit = torch.ge(y10, y11)
    tmp = torch.mul(bit, torch.tensor(128))
    val = torch.add(val, tmp)

    return val


map_pic = cv2.imread('data/NAIP/m_3809456_se_15_1_20090719.jpg')
PATCH_SIZE = 512
X = 500
Y = 500
patch = map_pic[
                Y - int(PATCH_SIZE / 2):Y + int(PATCH_SIZE / 2) + 1,
                X - int(PATCH_SIZE / 2):X + int(PATCH_SIZE / 2) + 1
                ]
plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
plt.title("Patch")
plt.axis('off')
plt.show()

gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # [H, W]
lbp_ski = local_binary_pattern(gray_patch, 8, 1, method='default')
lbp_ski = lbp_ski.astype(np.uint8)
gray_patch = gray_patch.astype(np.float32)
gray_tensor = torch.from_numpy(gray_patch).unsqueeze(0)
lbp_pt = lbp_torch(gray_tensor)

lbp_pt_np = lbp_pt.numpy().astype(np.uint8).flatten()
lbp_ski_flat = lbp_ski.flatten()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(lbp_pt_np, bins=256, range=(0, 255), color='steelblue')
plt.title("Histogram LBP (PyTorch)")
plt.xlabel("Wartość LBP")
plt.ylabel("Liczba pikseli")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(lbp_ski_flat, bins=256, range=(0, 255), color='darkorange')
plt.title("Histogram LBP (skimage, default)")
plt.xlabel("Wartość LBP")
plt.ylabel("Liczba pikseli")
plt.grid(True)

plt.tight_layout()
plt.show()
