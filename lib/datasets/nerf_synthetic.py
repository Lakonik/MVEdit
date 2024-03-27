import os
import io
import cv2
import json
import numpy
import random
# import trimesh
import matplotlib.pyplot as plotlib
from typing import Optional, List
from torch.utils.data import Dataset
from mmgen.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC
from multiprocessing.pool import ThreadPool


from .parallel_zip import ParallelZipFile as ZipFile

# b = numpy.random.randn(3, 32)
# x, y, z = b
# trimesh.transformations.affine_matrix_from_points(b, numpy.stack([x, z, -y]))
BLENDER_TO_OPENGL_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)
BLENDER_TO_OPENCV_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)


@DATASETS.register_module()
class NerfSynthetic(Dataset):

    def __init__(
        self, meta_files: list, world_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.meta_files = meta_files
        self.world_scale = world_scale

    def __len__(self):
        return len(self.meta_files)

    def load_sub(self, sub):
        with open(sub) as mf:
            meta = json.load(mf)
        frames_i = []
        frames_p = []
        frames_c = []
        for frame in range(len(meta['frames'])):
            img = plotlib.imread(os.path.join(os.path.dirname(sub), meta['frames'][frame]['file_path'] + '.png'))
            h, w, c = img.shape
            x, y = w / 2, h / 2
            focal_length = y / numpy.tan(meta['camera_angle_x'] / 2)
            # scaling = 320.0 / img.shape[0]
            scaling = 1.0
            img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])
            # img = cv2.resize(img, [320, 320], interpolation=cv2.INTER_AREA)
            pose = meta['frames'][frame]['transform_matrix']
            frames_i.append(img)
            frames_p.append((numpy.array(pose) @ BLENDER_TO_OPENCV_MATRIX) * self.world_scale)
            frames_c.append(numpy.array([focal_length, focal_length, x, y]) * scaling)
        f32 = numpy.float32
        return dict(
            cond_imgs=numpy.array(frames_i, f32),
            cond_poses=numpy.array(frames_p, f32),
            cond_intrinsics=numpy.array(frames_c, f32)
        )

    def __getitem__(self, index):
        sub = self.meta_files[index]
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(sub, cpu_only=True),
            **self.load_sub(sub)
        )
