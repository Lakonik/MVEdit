import os
import shutil
import trimesh
import argparse
import tqdm
from trimesh.visual.material import SimpleMaterial


def parse_args():
    parser = argparse.ArgumentParser(description='Convert GLB to OBJ')
    parser.add_argument('input', type=str, help='Input directory')
    parser.add_argument('output', type=str, help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    print(f'Converting GLB files from {args.input} to OBJ files in {out_dir}')
    in_files = [file for file in os.listdir(args.input) if file.endswith('.glb')
                and os.path.isfile(os.path.join(args.input, file))]
    for file in tqdm.tqdm(in_files):
        mesh = trimesh.load_mesh(os.path.join(args.input, file))
        assert isinstance(mesh, trimesh.Scene)
        for k, v in mesh.geometry.items():
            v.visual.material = SimpleMaterial(
                image=v.visual.material.baseColorTexture,
                diffuse=[255, 255, 255],
                ambient=[255, 255, 255],
                specular=[0, 0, 0],
                glossiness=0)
        out_dir_path = os.path.join(out_dir, file[:-4])
        if os.path.exists(out_dir_path):
            shutil.rmtree(out_dir_path)
        os.makedirs(out_dir_path)
        mesh.export(os.path.join(out_dir_path, 'mesh.obj'), include_normals=True)


if __name__ == '__main__':
    main()
