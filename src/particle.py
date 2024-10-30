import numpy as np


class Particle:
    def __init__(self, map_pic, patch_size: int):
        self.map_pic = map_pic
        self.height, self.width, self.canals = self.map_pic.shape
        self.patch_size = patch_size
        self.x = np.random.randint(
            0 + int(self.patch_size / 2) + 1,
            self.width - int(self.patch_size / 2) - 1,
            )
        self.y = np.random.randint(
            0 + int(self.patch_size / 2) + 1,
            self.height - int(self.patch_size / 2) - 1,
            )

        self.descriptor = None

    def descriptor_collect(self, map_pic):
        self.descriptor = map_pic[
            self.y - int(self.patch_size / 2):self.y + int(self.patch_size / 2) + 1,
            self.x - int(self.patch_size / 2):self.x + int(self.patch_size / 2) + 1
            ]

    def move(self):
        ...
