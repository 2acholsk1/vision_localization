import numpy as np

from logger import log


class UAV:
    def __init__(self, map_pic, patch_size: int):
        self.map_pic = map_pic
        self.height, self.width, self.canals = map_pic.shape
        self.patch_size = patch_size

        self.start_point = np.random.randint(
            0 + int(self.patch_size / 2) + 1,
            self.height - int(self.patch_size / 2) - 1,
            )
        self.end_point = np.random.randint(
            0 + int(self.patch_size / 2) + 1,
            self.height - int(self.patch_size / 2) - 1,
            )

        self.sequence_length = None
        self.localization: np.array = None
        self.traj_coords = np.array([])
        self.step = 0

        self.patch = None

        log.info(f'UAV start-> (0,{self.start_point}) and end-> ({self.width},{self.end_point}) points localization')

    def generate_trajectory(self, sequence_length):
        self.sequence_length = sequence_length
        coord_heights = np.linspace(
            self.start_point,
            self.end_point,
            self.sequence_length,
            )
        coord_widths = np.linspace(
            0 + int(self.patch_size / 2) + 1,
            self.width - int(self.patch_size / 2) - 1,
            sequence_length,
            )

        for i in range(self.sequence_length):
            self.traj_coords = np.append(
                    (
                        int(coord_widths[i]),
                        int(coord_heights[i]),
                    )
                )

    def set_patch(self):
        self.patch = self.map_pic[
            self.localization[1] - int(self.patch_size / 2):self.localization[1] + int(self.patch_size / 2) + 1,
            self.localization[0] - int(self.patch_size / 2):self.localization[0] + int(self.patch_size / 2) + 1
        ]

    def move(self):
        self.step += 1
        self.localization = self.traj_coords[self.step]
