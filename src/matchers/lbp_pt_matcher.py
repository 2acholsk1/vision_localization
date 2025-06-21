import cv2
import numpy as np
import torch
import torch.nn.functional as F


class MatcherPyTorchLBP:
    def __init__(self):
        self.template_vector = None
        self.sum_of_weight = []

    def match_patches(self, candidate):
        candidate_vector = self.compute_hist_descriptors(candidate)

        result = 1.0 - np.sum(abs(candidate_vector - self.template_vector))

        # Normalization between [0..1]
        result = (result + 1.0) / 2.0
        self.sum_of_weight.append(result)
        return result

    def compute_template(self, patch):
        self.template_vector = self.compute_hist_descriptors(patch)

    def get_sum_of_weight(self):
        sum_value = np.sum(self.sum_of_weight)
        self.sum_of_weight = []
        return sum_value

    def compute_hist_descriptors(self, patch):
        hist_b = cv2.calcHist([patch], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([patch], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([patch], [2], None, [16], [0, 256])

        hist_b /= hist_b.sum()
        hist_g /= hist_g.sum()
        hist_r /= hist_r.sum()

        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray_patch = gray_patch.astype(np.float32)
        gray_tensor = torch.from_numpy(gray_patch).unsqueeze(0)

        lbp = self.lbp_torch(gray_tensor)
        lbp = lbp.numpy().astype(np.uint8).flatten()

        hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 31])
        hist_lbp /= hist_lbp.sum()

        vector = np.vstack((hist_b, hist_g, hist_r, hist_lbp))
        vector /= np.sum(vector)
        return vector

    def lbp_torch(self, x):
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
