import os
import os.path as osp
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../../imagepacker/')))

import uuid
import tempfile
import tqdm
import math
import numpy as np
import cv2
import open3d as o3d
import torch
import trimesh
import torch.nn.functional as F
import pygltflib
import xatlas
import objuvpacker
import miniball
from collections import deque
from trimesh import util, resolvers
from trimesh.visual import TextureVisuals
from trimesh.visual.color import to_float
from PIL import Image


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


class Mesh:
    def __init__(
            self,
            v=None,
            f=None,
            vn=None,
            fn=None,
            vt=None,
            ft=None,
            vc=None,
            albedo=None,
            device=None,
            textureless=False):
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.vc = vc
        self.f = f
        self.fn = fn
        self.ft = ft
        self.face_normals = None
        # only support a single albedo
        self.albedo = albedo
        self.textureless = textureless

        self.ori_center = 0
        self.ori_scale = 1

    def detach(self):
        self.v = self.v.detach() if self.v is not None else None
        self.vn = self.vn.detach() if self.vn is not None else None
        self.vt = self.vt.detach() if self.vt is not None else None
        self.vc = self.vc.detach() if self.vc is not None else None
        self.f = self.f.detach() if self.f is not None else None
        self.fn = self.fn.detach() if self.fn is not None else None
        self.ft = self.ft.detach() if self.ft is not None else None
        self.face_normals = self.face_normals.detach() if self.face_normals is not None else None
        self.albedo = self.albedo.detach() if self.albedo is not None else None
        return self

    @classmethod
    def load(cls, path=None, resize=False, auto_uv=True, flip_yz=False,
             force_auto_normal=False, auto_normal_seamless=False, **kwargs):
        # assume init with kwargs
        if path is None:
            mesh = cls(**kwargs)
        # obj supports face uv
        elif path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs)

        print(f"[Mesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")
        # auto-normalize
        if resize:
            mesh.auto_size()
        # auto-fix normal
        if mesh.vn is None or force_auto_normal:
            mesh.auto_normal(seamless=auto_normal_seamless)
        print(f"[Mesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")
        # auto-fix texture
        if mesh.vt is None and auto_uv:
            mesh.auto_uv(cache_path=path)
        if mesh.vt is not None and mesh.ft is not None:
            print(f"[Mesh loading] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        if flip_yz:
            mesh.v[..., [1, 2]] = mesh.v[..., [2, 1]]
            mesh.vn[..., [1, 2]] = mesh.vn[..., [2, 1]]
            mesh.v[..., 1] = -mesh.v[..., 1]
            mesh.vn[..., 1] = -mesh.vn[..., 1]

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None):
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        mtl_path = None

        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            prefix = split_line[0].lower()
            # mtllib
            if prefix == "mtllib":
                mtl_path = split_line[1]
            # usemtl
            elif prefix == "usemtl":
                pass  # ignored
            # v/vn/vt
            elif prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if len(normals) > 0
            else None
        )

        # see if there is vertex color
        if mesh.v.size(-1) > 3:
            mesh.vc = mesh.v[:, 3:]
            mesh.v = mesh.v[:, :3]
            if mesh.vc.size(-1) == 3:
                mesh.vc = torch.cat([mesh.vc, torch.ones_like(mesh.vc[:, :1])], dim=-1)
            print(f"[load_obj] use vertex color: {mesh.vc.shape}")

        # try to retrieve mtl file
        mtl_path_candidates = []
        if mtl_path is not None:
            mtl_path_candidates.append(mtl_path)
            mtl_path_candidates.append(os.path.join(os.path.dirname(path), mtl_path))
        mtl_path_candidates.append(path.replace(".obj", ".mtl"))

        mtl_path = None
        for candidate in mtl_path_candidates:
            if os.path.exists(candidate):
                mtl_path = candidate
                break

        # if albedo_path is not provided, try retrieve it from mtl
        if mtl_path is not None and albedo_path is None:
            with open(mtl_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                split_line = line.split()
                # empty line
                if len(split_line) == 0:
                    continue
                prefix = split_line[0]
                # NOTE: simply use the first map_Kd as albedo!
                if "map_Kd" in prefix:
                    albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                    print(f"[load_obj] use texture from: {albedo_path}")
                    break

        # still not found albedo_path, or the path doesn't exist
        if albedo_path is None or not os.path.exists(albedo_path):
            # init an empty texture
            print(f"[load_obj] init empty albedo!")
            # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)
            albedo = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])  # default color
            mesh.textureless = True
        else:
            albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
            albedo = albedo.astype(np.float32) / 255
            print(f"[load_obj] load texture: {albedo.shape}")

            # import matplotlib.pyplot as plt
            # plt.imshow(albedo)
            # plt.show()

        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)

        return mesh

    @classmethod
    def load_trimesh(cls, path, device=None):
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load glb, assume only has one single RootMesh...
        _data = trimesh.load(path)
        if isinstance(_data, trimesh.Scene):
            mesh_keys = list(_data.geometry.keys())
            assert (
                    len(mesh_keys) == 1
            ), f"{path} contains more than one meshes, not supported!"
            _mesh = _data.geometry[mesh_keys[0]]

        elif isinstance(_data, trimesh.Trimesh):
            _mesh = _data

        else:
            raise NotImplementedError(f"type {type(_data)} not supported!")

        if hasattr(_mesh.visual, "material"):
            _material = _mesh.visual.material
            if isinstance(_material, trimesh.visual.material.PBRMaterial):
                texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
            elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
                texture = (
                        np.array(_material.to_pbr().baseColorTexture).astype(np.float32) / 255
                )
            else:
                raise NotImplementedError(f"material type {type(_material)} not supported!")

            print(f"[load_obj] load texture: {texture.shape}")
            mesh.albedo = torch.tensor(texture, dtype=torch.float32, device=device)

        if hasattr(_mesh.visual, "uv"):
            texcoords = _mesh.visual.uv
            texcoords[:, 1] = 1 - texcoords[:, 1]
            mesh.vt = (
                torch.tensor(texcoords, dtype=torch.float32, device=device)
                if len(texcoords) > 0
                else None
            )
        else:
            texcoords = None

        if hasattr(_mesh.visual, "vertex_colors"):
            colors = _mesh.visual.vertex_colors
            mesh.vc = (
                torch.tensor(colors, dtype=torch.float32, device=device) / 255
                if len(colors) > 0
                else None
            )

        vertices = _mesh.vertices

        normals = _mesh.vertex_normals

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    # aabb
    def aabb(self):
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 1.2 / torch.max(vmax - vmin).item()  # to ~ [-0.6, 0.6]
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self, seamless=False):
        if seamless:
            mesh_verts, indices = torch.unique(self.v, dim=0, return_inverse=True, sorted=False)
            mesh_faces = indices[self.f]
        else:
            mesh_verts = self.v
            mesh_faces = self.f

        i0, i1, i2 = mesh_faces[:, 0].long(), mesh_faces[:, 1].long(), mesh_faces[:, 2].long()
        v0, v1, v2 = mesh_verts[i0, :], mesh_verts[i1, :], mesh_verts[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        face_normals = F.normalize(face_normals, dim=-1)
        vn = torch.zeros_like(mesh_verts)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
        vn = F.normalize(vn, dim=-1)

        self.vn = vn
        self.fn = mesh_faces.to(torch.int32)
        self.face_normals = face_normals

    def auto_uv(self, cache_path=None, vmap=True):
        # try to load cache
        if cache_path is not None:
            cache_path = os.path.splitext(cache_path)[0] + '_uv.npz'

        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np, vmapping = data['vt'], data['ft'], data['vmapping']
        else:
            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np, vmapping=vmapping)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)
        self.vt = vt
        self.ft = ft

        if vmap:
            # remap v/f to vt/ft, so each v correspond to a unique vt. (necessary for gltf)
            vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
            self.align_v_to_vt(vmapping)

    def align_v_to_vt(self, vmapping=None):
        # remap v/f and vn/vn to vt/ft.
        if vmapping is None:
            ft = self.ft.view(-1).long()
            f = self.f.view(-1).long()
            vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = f  # scatter, randomly choose one if index is not unique
        if self.vc is not None:
            self.vc = self.vc[vmapping]
        if self.vn is not None and (self.f == self.fn).all():
            self.vn = self.vn[vmapping]
            self.fn = self.ft
        self.v = self.v[vmapping]
        self.f = self.ft

    def align_vn_to_vt(self, vmapping=None):
        if vmapping is None:
            ft = self.ft.view(-1).long()
            fn = self.f.view(-1).long()
            vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = fn  # scatter, randomly choose one if index is not unique
        self.vn = self.vn[vmapping]
        self.fn = self.ft

    def to(self, device):
        self.device = device
        for name in ['v', 'f', 'vn', 'fn', 'vt', 'ft', 'albedo', 'vc', 'face_normals']:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self

    def copy(self):
        return Mesh(
            v=self.v,
            f=self.f,
            vn=self.vn,
            fn=self.fn,
            vt=self.vt,
            ft=self.ft,
            vc=self.vc,
            albedo=self.albedo,
            device=self.device,
            textureless=self.textureless)

    def write(self, path, flip_yz=False):
        mesh = self.copy()
        if flip_yz:
            mesh.v = mesh.v.clone()
            mesh.vn = mesh.vn.clone()
            mesh.v[..., 1] = -mesh.v[..., 1]
            mesh.vn[..., 1] = -mesh.vn[..., 1]
            mesh.v[..., [1, 2]] = mesh.v[..., [2, 1]]
            mesh.vn[..., [1, 2]] = mesh.vn[..., [2, 1]]
        if path.endswith('.ply'):
            mesh.write_ply(path)
        elif path.endswith('.obj'):
            mesh.write_obj(path)
        elif path.endswith('.glb') or path.endswith('.gltf'):
            mesh.write_glb(path)
        else:
            raise NotImplementedError(f'format {path} not supported!')

    # write to ply file (only geom)
    def write_ply(self, path):

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)

    # write to gltf/glb file (geom + texture)
    def write_glb(self, path):

        assert self.vn is not None
        if self.vt is None:
            self.vt = self.v.new_zeros((self.v.size(0), 2))
            self.ft = self.f
        if (self.f != self.ft).any():
            self.align_v_to_vt()
        if (self.fn != self.ft).any():
            self.align_vn_to_vt()

        assert self.v.shape[0] == self.vn.shape[0] and self.v.shape[0] == self.vt.shape[0]

        f_np = self.f.detach().cpu().numpy().astype(np.uint32)
        v_np = self.v.detach().cpu().numpy().astype(np.float32)
        vt_np = self.vt.detach().cpu().numpy().astype(np.float32)
        vn_np = self.vn.detach().cpu().numpy().astype(np.float32)

        albedo = self.albedo.detach().cpu().numpy() if self.albedo is not None \
            else np.full((1024, 1024, 3), 0.5, dtype=np.float32)
        albedo = (albedo * 255).astype(np.uint8)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)

        f_np_blob = f_np.flatten().tobytes()
        v_np_blob = v_np.tobytes()
        vt_np_blob = vt_np.tobytes()
        vn_np_blob = vn_np.tobytes()
        albedo_blob = cv2.imencode('.png', albedo)[1].tobytes()

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[pygltflib.Mesh(primitives=[
                pygltflib.Primitive(
                    # indices to accessors (0 is triangles)
                    attributes=pygltflib.Attributes(
                        POSITION=1, TEXCOORD_0=2, NORMAL=3
                    ),
                    indices=0, material=0,
                )
            ])],
            materials=[
                pygltflib.Material(
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                    alphaCutoff=0,
                    doubleSided=True,
                )
            ],
            textures=[
                pygltflib.Texture(sampler=0, source=0),
            ],
            samplers=[
                pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR,
                                  wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT),
            ],
            images=[
                # use embedded (buffer) image
                pygltflib.Image(bufferView=4, mimeType="image/png"),
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(f_np_blob) + len(v_np_blob) + len(vt_np_blob) + len(vn_np_blob) + len(albedo_blob))
            ],
            # buffer view (based on dtype)
            bufferViews=[
                # triangles; as flatten (element) array
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(f_np_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,  # GL_ELEMENT_ARRAY_BUFFER (34963)
                ),
                # positions; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob),
                    byteLength=len(v_np_blob),
                    byteStride=12,  # vec3
                    target=pygltflib.ARRAY_BUFFER,  # GL_ARRAY_BUFFER (34962)
                ),
                # texcoords; as vec2 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob),
                    byteLength=len(vt_np_blob),
                    byteStride=8,  # vec2
                    target=pygltflib.ARRAY_BUFFER,
                ),
                # normals; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob) + len(vt_np_blob),
                    byteLength=len(vn_np_blob),
                    byteStride=12,  # vec3
                    target=pygltflib.ARRAY_BUFFER,
                ),
                # texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob) + len(vt_np_blob) + len(vn_np_blob),
                    byteLength=len(albedo_blob),
                ),
            ],
            accessors=[
                # 0 = triangles
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT,  # GL_UNSIGNED_INT (5125)
                    count=f_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(f_np.max())],
                    min=[int(f_np.min())],
                ),
                # 1 = positions
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,  # GL_FLOAT (5126)
                    count=len(v_np),
                    type=pygltflib.VEC3,
                    max=v_np.max(axis=0).tolist(),
                    min=v_np.min(axis=0).tolist(),
                ),
                # 2 = texcoords
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(vt_np),
                    type=pygltflib.VEC2,
                    max=vt_np.max(axis=0).tolist(),
                    min=vt_np.min(axis=0).tolist(),
                ),
                # 3 = normals
                pygltflib.Accessor(
                    bufferView=3,
                    componentType=pygltflib.FLOAT,
                    count=len(vn_np),
                    type=pygltflib.VEC3,
                    max=vn_np.max(axis=0).tolist(),
                    min=vn_np.min(axis=0).tolist(),
                ),
            ],
        )

        # set actual data
        gltf.set_binary_blob(f_np_blob + v_np_blob + vt_np_blob + vn_np_blob + albedo_blob)

        # glb = b"".join(gltf.save_to_bytes())
        gltf.save(path)

    # write to obj file (geom + texture)
    def write_obj(self, path):

        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            for v in v_np:
                fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} \n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]:.4f} {1 - v[1]:.4f} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                )

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            if not self.textureless and self.albedo is not None:
                fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")

        if not self.textureless and self.albedo is not None:
            albedo = self.albedo.detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))


def normalize_mesh(mesh, tgt_radius=0.9):
    mb = miniball.Miniball(mesh.v[:, :2].cpu().numpy())
    center_xy = mb.center()
    radius_xy_sq = mb.squared_radius()
    max_z = mesh.v[:, 2].max().item()
    min_z = mesh.v[:, 2].min().item()
    center = mesh.v.new_tensor([center_xy[0], center_xy[1], (max_z + min_z) / 2])
    radius = max(math.sqrt(radius_xy_sq), (max_z - min_z) / 2)
    scale = tgt_radius / radius
    mesh.v = (mesh.v - center) * scale
    return mesh, center.tolist(), scale


def check_has_texture_single(geom):
    return isinstance(geom.visual, TextureVisuals) and geom.visual.material.baseColorTexture is not None


def check_has_texture(mesh):
    if isinstance(mesh, trimesh.Scene):
        has_texture = []
        for geom in mesh.geometry.values():
            has_texture.append(check_has_texture_single(geom))
    elif isinstance(mesh, trimesh.Trimesh):
        has_texture = check_has_texture_single(mesh)
    else:
        raise NotImplementedError(f"type {type(mesh)} not supported!")
    return has_texture


def create_texture(geom):
    if hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'main_color'):
        main_color = tuple(geom.visual.material.main_color)
    else:
        main_color = (128, 128, 128)
    geom.visual = trimesh.visual.TextureVisuals(
        uv=np.full((geom.vertices.shape[0], 2), 0.5),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new('RGB', (8, 8), main_color)
        )
    )


def color_to_texture(mesh):
    if isinstance(mesh, trimesh.Scene):
        for geom in mesh.geometry.values():
            if not check_has_texture_single(geom):
                create_texture(geom)
    elif isinstance(mesh, trimesh.Trimesh):
        if not check_has_texture_single(mesh):
            create_texture(mesh)
    else:
        raise NotImplementedError(f"type {type(mesh)} not supported!")
    return mesh


def purge_scene(scene):
    update_flag = False
    delete_list = []
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            update_flag = True
            delete_list.append(name)
    for name in delete_list:
        scene.delete_geometry(name)
    return update_flag


def trimesh_scene_to_obj(
        scene,
        file_obj,
        include_normals=None,
        include_color=True,
        include_texture=True,
        write_texture=True,
        digits=8,
        mtl_name=None):
    resolver = resolvers.FilePathResolver(file_obj)

    face_formats = {
        ("v",): "{}",
        ("v", "vn"): "{}//{}",
        ("v", "vt"): "{}/{}",
        ("v", "vn", "vt"): "{}/{}/{}",
    }

    # convert scene to meshes
    meshes = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        # get a copy of the geometry
        current = scene.geometry[geometry_name]
        if hasattr(current, 'vertex_normals'):
            normals = np.array(current.vertex_normals)
            current = current.copy()
            current.vertex_normals = normals
        else:
            current = current.copy()

        # if the geometry is 2D see if we have to upgrade to 3D
        if hasattr(current, "to_3D"):
            # check to see if the scene is transforming the path out of plane
            check = util.isclose(transform, util._IDENTITY, atol=1e-8)
            check[:2, :3] = True
            if not check.all():
                # transform moves in 3D so we put this on the Z=0 plane
                current = current.to_3D()
            else:
                # transform moves in 2D so clip off the last row and column
                transform = transform[:3, :3]

        # move the geometry vertices into the requested frame
        current.apply_transform(transform)

        current.metadata["name"] = geometry_name
        current.metadata["node"] = node_name

        # save to our list of meshes
        meshes.append(current)

    # collect lines to export
    objects = deque([])
    # keep track of the number of each export element
    counts = {"v": 0, "vn": 0, "vt": 0}
    # collect materials as we go
    materials = {}
    materials_name = set()

    for current in meshes:
        # we are going to reference face_formats with this
        face_type = ["v"]
        # OBJ includes vertex color as RGB elements on the same line
        if (
                include_color
                and current.visual.kind in ["vertex", "face"]
                and len(current.visual.vertex_colors)
        ):
            # create a stacked blob with position and color
            v_blob = np.column_stack(
                (current.vertices, to_float(current.visual.vertex_colors[:, :3]))
            )
        else:
            # otherwise just export vertices
            v_blob = current.vertices

            # add the first vertex key and convert the array
        # add the vertices
        export = deque(
            [
                "v "
                + util.array_to_string(
                    v_blob, col_delim=" ", row_delim="\nv ", digits=digits
                )
            ]
        )

        # if include_normals is None then
        # only include if they're already stored
        if include_normals is None:
            include_normals = "vertex_normals" in current._cache.cache

        if include_normals:
            converted = util.array_to_string(
                current.vertex_normals,
                col_delim=" ",
                row_delim="\nvn ",
                digits=digits,
            )
            # if vertex normals are stored in cache export them
            face_type.append("vn")
            export.append("vn " + converted)

        # collect materials into a dict
        if include_texture and hasattr(current.visual, "uv"):
            # get a SimpleMaterial
            material = current.visual.material
            if hasattr(material, "to_simple"):
                material = material.to_simple()

            # hash the material to avoid duplicates
            hashed = hash(material)
            if hashed not in materials:
                # get a unique name for the material
                name = util.unique_name(material.name, materials_name)
                # add the name to our collection
                materials_name.add(name)
                # convert material to an OBJ MTL
                materials[hashed] = material.to_obj(name=name)

            # get the name of the current material as-stored
            tex_name = materials[hashed][1]

            # export the UV coordinates
            if len(np.shape(getattr(current.visual, "uv", None))) == 2:
                converted = util.array_to_string(
                    current.visual.uv, col_delim=" ", row_delim="\nvt ", digits=digits
                )
                # if vertex texture exists and is the right shape
                face_type.append("vt")
                # add the uv coordinates
                export.append("vt " + converted)
            # add the directive to use the exported material
            export.appendleft(f"usemtl {tex_name}")

        # the format for a single vertex reference of a face
        face_format = face_formats[tuple(face_type)]
        # add the exported faces to the export if available
        if hasattr(current, "faces"):
            export.append(
                "f "
                + util.array_to_string(
                    current.faces + 1 + counts["v"],
                    col_delim=" ",
                    row_delim="\nf ",
                    value_format=face_format,
                )
            )
        # offset our vertex position
        counts["v"] += len(current.vertices)

        # add object name if found in metadata
        if "name" in current.metadata:
            export.appendleft("\no {}".format(current.metadata["name"]))
        # add this object
        objects.append("\n".join(export))

    # collect files like images to write
    mtl_data = {}
    # combine materials
    if len(materials) > 0:
        # collect text for a single mtllib file
        mtl_lib = []
        # now loop through: keys are garbage hash
        # values are (data, name)
        for data, _ in materials.values():
            for file_name, file_data in data.items():
                if file_name.lower().endswith(".mtl"):
                    # collect mtl lines into single file
                    mtl_lib.append(file_data)
                elif file_name not in mtl_data:
                    # things like images
                    mtl_data[file_name] = file_data
                else:
                    print(f"not writing {file_name}")

        if mtl_name is None:
            # if no name passed set a default
            mtl_name = "material.mtl"

        # prepend a header to the MTL text if requested
        prepend = b""

        # save the material data
        mtl_data[mtl_name] = prepend + b"\n\n".join(mtl_lib)
        # add the reference to the MTL file
        objects.appendleft(f"mtllib {mtl_name}")


    # add a trailing newline
    objects.append("\n")

    # combine elements into a single string
    data = "\n".join(objects)

    # if we have a resolver and have asked to write texture
    if write_texture and resolver is not None and len(materials) > 0:
        # not all resolvers have a write method
        [resolver.write(k, v) for k, v in mtl_data.items()]

    file_path = os.path.abspath(os.path.expanduser(file_obj))
    with open(file_path, "wb") as f:
        util.write_encoded(f, data)

    return


def preprocess_mesh(in_mesh_path, normalize=True, cache_dir=None):
    if cache_dir is None:
        cache_dir = tempfile.gettempdir()
    tmp_dir = osp.join(cache_dir, str(uuid.uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)

    in_mesh = trimesh.load(in_mesh_path)

    mesh_requires_update = purge_scene(in_mesh) or (not in_mesh_path.endswith('.obj'))
    has_texture = check_has_texture(in_mesh)
    is_multi_object = isinstance(in_mesh, trimesh.Scene) and len(in_mesh.geometry) > 1
    if is_multi_object and not all(has_texture):
        in_mesh = color_to_texture(in_mesh)
        mesh_requires_update = True

    if mesh_requires_update:
        in_mesh_path = osp.join(tmp_dir, 'in_mesh.obj')
        if isinstance(in_mesh, trimesh.Scene):  # using trimesh export will reset the normals...
            trimesh_scene_to_obj(in_mesh, in_mesh_path, include_normals=True)
        else:
            in_mesh.export(in_mesh_path, include_normals=True)

    out_dict = dict()

    if normalize or is_multi_object:
        if is_multi_object:
            packed_mesh_path = osp.join(tmp_dir, 'packed_mesh.obj')
            ori_dir = osp.abspath(os.getcwd())
            sys.argv = ['',
                        osp.abspath(in_mesh_path),
                        '-o', osp.abspath(packed_mesh_path)[:-4],
                        '--no-crop']
            objuvpacker.main()
            os.chdir(ori_dir)

            packed_mesh = Mesh.load(packed_mesh_path, flip_yz=True, device='cpu')
            in_height, in_width = packed_mesh.albedo.shape[:2]
            out_width = 2 ** math.ceil(math.log2(in_width))
            out_height = 2 ** math.ceil(math.log2(in_height))
            pad_width = out_width - in_width
            pad_height = out_height - in_height
            packed_mesh.albedo = F.pad(packed_mesh.albedo, (0, 0, 0, pad_width, 0, pad_height), 'constant', 0.0)
            packed_mesh.vt[:, 0] *= in_width / out_width
            packed_mesh.vt[:, 1] *= in_height / out_height
        else:
            packed_mesh = Mesh.load(in_mesh_path, flip_yz=True, device='cpu')

        if normalize:
            packed_mesh, center, scale = normalize_mesh(packed_mesh)
            out_dict.update(center=center, scale=scale)

        proc_mesh_path = osp.join(tmp_dir, 'proc_mesh.obj')
        packed_mesh.write(proc_mesh_path, flip_yz=True)

        out_dict.update(mesh_path=proc_mesh_path, mesh_obj=packed_mesh)

    else:
        out_dict.update(mesh_path=in_mesh_path)

    return out_dict


def tsdf_rgbd_to_mesh(
        rgbs, depths, poses, intrinsics,
        bound=1.0, voxel_resolution=352, prune_thr=800,
        mesh_reduction=0.2, depth_trunc=10.0):
    """
    Args:
        rgbs (np.ndarray): (N, H, W, 3); float32
        depths (np.ndarray): (N, H, W); float32
        poses (np.ndarray): (N, 4, 4) C2W; float32
        intrinsics (np.ndarray): (N, 4) in [fx, fy, cx, cy]
        out_mesh_fpath (string): path to obj file
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=bound * 2 / voxel_resolution,
        sdf_trunc=bound * 2 * 0.02,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    rgbs = np.round(rgbs.clip(min=0, max=1) * 255).astype(np.uint8)
    w2cs = np.linalg.inv(poses)
    n, h, w = depths.shape
    for rgb, depth, w2c, intrinsic in tqdm.tqdm(zip(rgbs, depths, w2cs, intrinsics), total=n):
        rgb = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb, depth=depth, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        cam_intrinsics.set_intrinsics(
            w, h, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
        volume.integrate(rgbd, cam_intrinsics, w2c)

    mesh = volume.extract_triangle_mesh()

    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < prune_thr
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    # mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh = mesh.simplify_quadric_decimation(
        round(len(mesh.triangles) * mesh_reduction), boundary_weight=0).compute_vertex_normals()
    faces = torch.from_numpy(np.asarray(mesh.triangles)).long()

    out_mesh = Mesh(
        v=torch.from_numpy(np.asarray(mesh.vertices)).float(),
        vn=torch.from_numpy(np.asarray(mesh.vertex_normals)).float(),
        vc=torch.from_numpy(np.asarray(mesh.vertex_colors)).float(),
        f=faces, fn=faces)

    return out_mesh
