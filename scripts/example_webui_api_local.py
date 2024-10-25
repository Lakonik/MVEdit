import os
import shutil
from gradio_client import Client

out_dir = '../exp'
os.makedirs(out_dir, exist_ok=True)

client = Client('http://127.0.0.1:7860/')
seed = 42

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
