import numpy as np

from src.logger import log


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
        self.traj_coords = []
        self.step = 0

        self.patch = None
        self.move_diff = None
        log.info(
            "UAV start-> (0,%d) and end-> (%d,%d) points localization",
            self.start_point,
            self.width,
            self.end_point
            )


    def generate_trajectory(self, traj_type: str, sequence_length: int, amplitude_conf: int=250, freq_conf: int=3):
        self.sequence_length = sequence_length

        match traj_type:
            case 'simple':
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
            case 'sinusoidal':
                amplitude = amplitude_conf
                frequency = freq_conf
                offset = self.start_point

                coord_widths = np.linspace(
                    0 + int(self.patch_size / 2) + 1,
                    self.width - int(self.patch_size / 2) - 1,
                    self.sequence_length,
                )

                x_vals = np.linspace(0, 2 * np.pi * frequency, self.sequence_length)
                coord_heights = amplitude * np.sin(x_vals) + offset
            case _:
                coord_heights = []
                coord_widths = []

        self.traj_coords = []

        for i in range(self.sequence_length):
            self.traj_coords.append(
                (
                    int(coord_widths[i]),
                    int(coord_heights[i])
                )
            )

        self.localization = self.traj_coords[0]

    def set_patch(self):
        self.patch = self.map_pic[
            self.localization[1] - int(self.patch_size / 2):self.localization[1] + int(self.patch_size / 2) + 1,
            self.localization[0] - int(self.patch_size / 2):self.localization[0] + int(self.patch_size / 2) + 1
        ]

    def move(self):
        if self.step < (len(self.traj_coords)-1):
            self.step += 1
            self.localization = self.traj_coords[self.step]
            self.move_diff = np.array(self.traj_coords[self.step]) - np.array(self.traj_coords[self.step-1])
            return False
        return True

    def get_position(self):
        return self.localization

    def get_patch(self):
        return self.patch
