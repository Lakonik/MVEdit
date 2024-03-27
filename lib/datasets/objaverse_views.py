import os
import io
import cv2
import json
import gzip
import pickle
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


def smart_crop(img_in, min_size: int, crop_ratio: float = 0.9):
    # img_in: s, s, c
    uncroppable = numpy.argwhere((img_in != img_in[0, 0]).all(-1))
    if not len(uncroppable):
        return 0, img_in
    max_crop = max(int(len(img_in) // 2 - (len(img_in) // 2 - uncroppable.min()) / crop_ratio - 2), 0)
    max_crop = min(len(img_in) - int(len(img_in) // 2 + (uncroppable.max() - len(img_in) // 2) / crop_ratio - 2), max_crop)
    max_crop = min(max_crop, (len(img_in) - min_size) // 2)
    if max_crop <= 0:
        return 0, img_in
    return max_crop, img_in[max_crop: -max_crop, max_crop: -max_crop]


@DATASETS.register_module()
class ObjaverseViews(Dataset):
    archives: Optional[List[ZipFile]]

    def __init__(
        self, archive_paths, max_frames: int, memcache_all: bool, world_scale: float = 1.0, smart_crop_res=256,
        code_dir=None
    ) -> None:
        super().__init__()
        self.archives = None
        self.archive_paths = archive_paths
        self.max_frames = max_frames
        self.memcache_all = memcache_all
        self.world_scale = world_scale
        self.smart_crop_res = smart_crop_res
        self.code_dir = code_dir
        with gzip.open('captions.pkl.gz') as fi:
            self.captions: dict = pickle.load(fi)

    def lazy_init_unpicklable(self):
        if self.archives is None:
            self.archives = [ZipFile(p) for p in self.archive_paths]
            self.submap = {os.path.split(u)[0]: archive for archive in self.archives for u in archive.namelist()}
            self.subs = sorted(self.submap.keys()) if self.code_dir is None else [os.path.splitext(x)[0] for x in sorted(os.listdir(self.code_dir))]
            for sub in self.subs:
                assert sub in self.submap, sub
            self.cache = dict()
            if self.memcache_all:
                with ThreadPool(8) as pool:
                    for sub, data in zip(self.subs, pool.map(self.load_sub, self.subs)):
                        self.cache[sub] = data  # self.load_sub(sub)

    def __len__(self):
        self.lazy_init_unpicklable()
        return len(self.subs)

    def load_sub(self, sub):
        if sub in self.cache:
            return self.cache[sub]
        archive = self.submap[sub]
        with io.BytesIO(archive.read(f"{sub}/meta.json")) as fi:
            meta = json.load(fi)
        frames_i = []
        frames_p = []
        frames_c = []
        choice_ic = random.randrange(self.max_frames)
        frames_ic = None
        pose_ic = None
        for frame in range(self.max_frames):
            with io.BytesIO(archive.read(f"{sub}/rgb_{frame:04}.png")) as fi:
                img = plotlib.imread(fi, "png")
            if choice_ic == frame:
                frames_ic = ((img[..., :3] * img[..., 3:] + (1 - img[..., 3:])) * 2.0 - 1.0) * 0.6
                frames_ic = cv2.resize(frames_ic, [384, 384], interpolation=cv2.INTER_LINEAR)
            h, w, c = img.shape
            x, y = w / 2, h / 2
            focal_length = y / numpy.tan(meta['fovy'] / 2)
            shift, img = smart_crop(img, self.smart_crop_res)
            scaling = 320.0 / img.shape[0]
            img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])
            img = cv2.resize(img, [320, 320], interpolation=cv2.INTER_AREA)
            pose = meta['frames'][frame]['transform_matrix']
            frames_i.append(img)
            frames_p.append((numpy.array(pose) @ BLENDER_TO_OPENCV_MATRIX) * self.world_scale)
            if choice_ic == frame:
                pose_ic = ((numpy.array(pose) @ BLENDER_TO_OPENCV_MATRIX) * self.world_scale)[:3].reshape(12)
            frames_c.append(numpy.array([focal_length, focal_length, x - shift, y - shift]) * scaling)
        f32 = numpy.float32
        return dict(
            cond_imgs=numpy.array(frames_i, f32),
            cond_poses=numpy.array(frames_p, f32),
            cond_intrinsics=numpy.array(frames_c, f32),
            extra_cond_img=numpy.array(frames_ic, f32),
            extra_pose_cond=numpy.array(pose_ic, f32),
            prompts=DC(random.choice(self.captions.get(sub, [''])), cpu_only=True)
        )

    def __getitem__(self, index):
        self.lazy_init_unpicklable()
        sub = self.subs[index]
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(sub, cpu_only=True),
            **self.load_sub(sub)
        )
