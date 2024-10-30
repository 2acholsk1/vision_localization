import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class MatcherLBP:
    def __init__(self):
        self.template_descriptor = None

    def match_patches(self, candidate):
        candidate_descriptor = self.compute_hist_descriptors(candidate)
        candidate_descriptor /= np.sum(candidate_descriptor)

        result = 1.0 - np.sum(np.abs(candidate_descriptor - self.template_descriptor))
        return result

    def compute_template_descriptor(self, patch):
        self.template_descriptor = self.compute_hist_descriptors(patch)

    @staticmethod
    def compute_hist_descriptors(patch):
        hist_b = cv2.calcHist([patch], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([patch], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([patch], [2], None, [16], [0, 256])

        hist_b, hist_g, hist_r = hist_b / hist_b.sum(), hist_g / hist_g.sum(), hist_r / hist_r.sum()

        lbp = local_binary_pattern(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 8, 1, method='nri_uniform')
        hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 31])
        hist_lbp = hist_lbp / hist_lbp.sum()

        return np.vstack((hist_b, hist_g, hist_r, hist_lbp))
