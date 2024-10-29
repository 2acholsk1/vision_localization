
class Particle:
    def __init__(self, map_pic, patch_size: int):
        self.map_pic = map_pic
        self.patch_size = patch_size

    def descriptor_collect(self):
        ...

    def move(self):
        ...
