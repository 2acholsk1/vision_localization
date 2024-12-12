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

        self.patch = None
        self.weight = None
        self.x_new = None
        self.y_new = None

    def set_patch(self):
        self.patch = self.map_pic[
            self.y - int(self.patch_size / 2):self.y + int(self.patch_size / 2) + 1,
            self.x - int(self.patch_size / 2):self.x + int(self.patch_size / 2) + 1
            ]

    def xy_new_swap(self):
        self.x = self.x_new
        self.y = self.y_new

    def move(self, rand_val_move, uav_move):
        rand_movement = np.random.randint(-rand_val_move, rand_val_move, size=uav_move.shape)
        total_move = rand_movement + uav_move
        self.x += total_move[0]
        self.y += total_move[1]
        self.x = np.clip(self.x, 0 + int(self.patch_size / 2) + 1,
                         self.width - int(self.patch_size / 2) - 1)
        self.y = np.clip(self.y, 0 + int(self.patch_size / 2) + 1,
                         self.height - int(self.patch_size / 2) - 1)

    def get_position(self):
        return [self.x, self.y]

    def get_patch(self):
        return self.patch
