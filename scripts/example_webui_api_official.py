import os
import shutil
from gradio_client import Client

out_dir = '../exp'
os.makedirs(out_dir, exist_ok=True)

client = Client('https://mvedit.hanshengchen.com/')
seed = 42

# ======== Text-to-3D using GRM Adapter (Instant3D) ========
prompt = 'a bichon frise wearing academic regalia'

grm_adapter_result = client.predict(
    seed,
    prompt,  # 'Prompt' Textbox component
    '',  # 'Negative prompt' Textbox component
    'EulerAncestralDiscrete',  # Sampling method Dropdown component
    30,  # 'Sampling steps' Slider component
    1.0,  # 'Denoising strength' Slider component
    False,  # 'Random initialization' Checkbox component
    5.0,  # 'CFG scale' Slider component
    2.0,  # 'Adapter scale' Slider component
    api_name='/text_to_3d_grm_adapter')
text_to_3d_gs_path = out_dir + '/text_to_3d_gs.ply'
shutil.move(grm_adapter_result, text_to_3d_gs_path)

tsdf_result = client.predict(
    text_to_3d_gs_path,
    352,  # 'Voxel resolution' Slider component
    800,  # 'Pruning threshold (# of triangles)' Slider component
    0.1,  # 'Decimation ratio' Slider component
    True,  # 'Fill bottom' Checkbox component
    api_name='/gs_to_mesh_tsdf')
text_to_3d_tsdf_path = out_dir + '/text_to_3d_tsdf.glb'
shutil.move(tsdf_result, text_to_3d_tsdf_path)

preproc_result = client.predict(
    text_to_3d_tsdf_path,
    api_name='/3d_preproc')[1]
superres_result = client.predict(
    seed,
    preproc_result,
    None,  # 'Front view ID' Number component
    'Default',	# 'Camera set' Dropdown component
    prompt,  # 'Prompt' Textbox component
    '',  # 'Negative prompt' Textbox component
    'EulerAncestralDiscrete',  # Sampling method Dropdown component
    24,  # 'Sampling steps' Slider component
    0.0,  # 'Denoising strength' Slider component
    False,  # 'Random initialization' Checkbox component
    7.0,  # 'CFG scale' Slider component
    False,  # 'Force auto UV' Checkbox component
    True,  # 'Texture super-resolution' Checkbox component
    False,  # (Texture super-resolution) 'Use IP-Adapter' Checkbox component
    'EulerAncestralDiscrete',  # (Texture super-resolution) 'Sampling method' Dropdown component
    24,  # (Texture super-resolution) 'Sampling steps' Slider component
    0.4,  # (Texture super-resolution) 'Denoising strength' Slider component
    False,  # (Texture super-resolution) 'Random initialization' Checkbox component
    7,  # (Texture super-resolution) 'CFG scale' Slider component
    None,  # (Texture super-resolution) 'Input image' Image component
    api_name='/retex_mvedit')
text_to_3d_path = out_dir + '/text_to_3d.glb'
shutil.move(superres_result, text_to_3d_path)

# ======== Image-to-3D using GRM Adapter (Zero123++ v1.2) ========
in_img = 'https://raw.githubusercontent.com/Lakonik/MVEdit/refs/heads/main/demo/examples_images/turtle.png'

seg_result = client.predict(
    in_img,
    api_name='/image_segmentation')

grm_adapter_result = client.predict(
    seed,
    seg_result,
    'EulerAncestralDiscrete',  # Sampling method Dropdown component
    32,  # 'Sampling steps' Slider component
    4,  # 'CFG scale' Slider component
    2.0,  # 'Adapter scale' Slider component
    api_name='/img_to_3d_grm_adapter')
img_to_3d_gs_path = out_dir + '/img_to_3d_gs.ply'
shutil.move(grm_adapter_result, img_to_3d_gs_path)

tsdf_result = client.predict(
    img_to_3d_gs_path,
    'TSDF',  # 'GS to mesh method' Dropdown component
    352,  # 'Voxel resolution' Slider component
    800,  # 'Pruning threshold (# of triangles)' Slider component
    0.1,  # 'Decimation ratio' Slider component
    True,  # 'Fill bottom' Checkbox component
    512,  # 'Optimization steps' Slider component
    api_name='/img_to_3d_grm_adapter_to_mesh')
img_to_3d_tsdf_path = out_dir + '/img_to_3d_tsdf.glb'
shutil.move(tsdf_result, img_to_3d_tsdf_path)

superres_result = client.predict(
    seed,
    seg_result,
    img_to_3d_gs_path,
    img_to_3d_tsdf_path,
    True,  # 'Texture super-resolution' Checkbox component
    True,  # 'Use IP-Adapter' Checkbox component
    'DPMSolverSDEKarras',  # 'Sampling method' Dropdown component
    24,  # 'Sampling steps' Slider component
    0.4,  # 'Denoising strength' Slider component
    False,  # 'Random initialization' Checkbox component
    7,  # 'CFG scale' Slider component
    api_name='/img_to_3d_grm_adapter_superres')
img_to_3d_path = out_dir + '/img_to_3d.glb'
shutil.move(superres_result, img_to_3d_path)

# ======== Text-to-texture using MVEdit ========
prompt = 'an nba basketball player'
in_mesh = 'https://raw.githubusercontent.com/Lakonik/MVEdit/refs/heads/main/demo/examples_meshes/cj_notex.glb'
front_view_id = 9

preproc_result = client.predict(
    in_mesh,
    api_name='/3d_preproc')[1]

result = client.predict(
    seed,
    preproc_result,
    front_view_id,
    'Default',  # 'Camera set' Dropdown component
    prompt,
    '',	 # 'Negative prompt' Textbox component
    'EulerAncestralDiscrete',  # Sampling method Dropdown component
    24,	 # 'Sampling steps' Slider component
    0.9,  # 'Denoising strength' Slider component
    False,	# 'Random initialization' Checkbox component
    7,	# 'CFG scale' Slider component
    True,	# 'Force auto UV' Checkbox component
    False,	# 'Texture super-resolution' Checkbox component
    False,  # (Texture super-resolution) 'Use IP-Adapter' Checkbox component
    'EulerAncestralDiscrete',  # (Texture super-resolution) 'Sampling method' Dropdown component
    24,	 # (Texture super-resolution) 'Sampling steps' Slider component
    0.5,  # (Texture super-resolution) 'Denoising strength' Slider component
    False,	# (Texture super-resolution) 'Random initialization' Checkbox component
    7,	# (Texture super-resolution) 'CFG scale' Slider component
    None,  # (Texture super-resolution) 'Input image' Image component
    api_name='/retex_mvedit')
text_to_tex_path = out_dir + '/text_to_tex.glb'
shutil.move(result, text_to_tex_path)
