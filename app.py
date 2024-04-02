import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../')))
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '16'

import shutil
import os.path as osp
import argparse
import torch
import gradio as gr
from functools import partial
from lib.core.mvedit_webui.shared_opts import send_to_click
from lib.core.mvedit_webui.tab_img_to_3d import create_interface_img_to_3d
from lib.core.mvedit_webui.tab_3d_to_3d import create_interface_3d_to_3d
from lib.core.mvedit_webui.tab_text_to_img_to_3d import create_interface_text_to_img_to_3d
from lib.core.mvedit_webui.tab_retexturing import create_interface_retexturing
from lib.core.mvedit_webui.tab_3d_to_video import create_interface_3d_to_video
from lib.core.mvedit_webui.tab_stablessdnerf_to_3d import create_interface_stablessdnerf_to_3d
from lib.apis.mvedit import MVEditRunner


def parse_args():
    parser = argparse.ArgumentParser(description='MVEdit 3D Toolbox')
    parser.add_argument('--diff-bs', type=int, default=4, help='Diffusion batch size')
    parser.add_argument('--advanced', action='store_true', help='Show advanced settings')
    parser.add_argument('--debug', action='store_true', help='Save debug images to ./viz')
    parser.add_argument('--local-files-only', action='store_true',
                        help='Only load local model weights and configuration files')
    parser.add_argument('--no-safe', action='store_true', help='Disable safety checker to free VRAM')
    parser.add_argument('--empty-cache', action='store_true', help='Empty the cache directory')
    parser.add_argument('--unload-models', action='store_true', help='Auto-unload unused models to free VRAM')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.empty_cache:
        if osp.exists('./gradio_cached_examples'):
            shutil.rmtree('./gradio_cached_examples')
        if os.environ.get('GRADIO_TEMP_DIR', None) is not None and osp.exists(os.environ['GRADIO_TEMP_DIR']):
            shutil.rmtree(os.environ['GRADIO_TEMP_DIR'])

    torch.set_grad_enabled(False)
    runner = MVEditRunner(
        device=torch.device('cuda'),
        local_files_only=args.local_files_only,
        unload_models=args.unload_models,
        out_dir=osp.join(osp.dirname(__file__), 'viz') if args.debug else None,
        save_interval=1 if args.debug else None,
        debug=args.debug,
        no_safe=args.no_safe
    )

    with gr.Blocks(analytics_enabled=False,
                   title='MVEdit 3D Toolbox',
                   css='lib/core/mvedit_webui/style.css'
                   ) as demo:
        md_txt = '# MVEdit 3D Toolbox' \
                 '\n\nOfficial demo of the paper [Generic 3D Diffusion Adapter Using Controlled Multi-View Editing](https://lakonik.github.io/mvedit).' \
                 '<br>The generation process is stochastic so the results vary between runs. ' \
                 'Try multiple runs if you find the result unsatisfactory.' \
                 '<br>MVEdit adopts Stable Diffusion v1.5 models. Finetuned community ' \
                 'checkpoints can be applied in the **Advanced settings**. These models can ' \
                 'reinforce or exacerbate social biases. Misuse or malicious use is prohibited.'
        if not args.advanced:
            md_txt += '<br>**Advanced settings** are disabled. Deploy the app with `--advanced` to enable them.'
        gr.Markdown(md_txt)

        with gr.Tabs(selected='tab_img_to_3d') as main_tabs:
            with gr.TabItem('Text-to-3D', id='tab_text_to_3d'):
                with gr.Tabs() as sub_tabs_text_to_3d:
                    with gr.TabItem('StableSSDNeRF (ShapeNet Cars)', id='tab_stablessdnerf'):
                        _, var_stablessdnerf = create_interface_stablessdnerf_to_3d(
                            runner.run_stablessdnerf,
                            runner.run_stablessdnerf_to_mesh,
                            examples=[
                                ['a lego volkswagon beetle', ''],
                                ['a lego bugatti veyron', ''],
                                ['a formula 1 racing car', ''],
                                ['game ready 3d model of a racing truck', ''],
                                ['a rusty old car', ''],
                                ['a modified wide-body racing limo', ''],
                                ['a ferrari 458 gt3 racing car', ''],
                                ['game ready 3d model of a porsche 911 police car, police light bar', ''],
                                ['game ready 3d model of a cyberpunk big wheel monster truck', ''],
                                ['a futuristic racing car', '']
                            ],
                            api_names=['text_to_3d_stablessdnerf', 'text_to_3d_stablessdnerf_to_mesh'],
                            diff_bs=args.diff_bs, advanced=args.advanced)
                    with gr.TabItem('Text-to-Image-to-3D', id='tab_text_to_img_to_3d'):
                        _, var_text_to_img_to_3d = create_interface_text_to_img_to_3d(
                            runner.run_text_to_img,
                            examples=[
                                [768, 512, 'a wooden carving of a wise old turtle', ''],
                                [512, 512, 'a glowing robotic unicorn, full body', ''],
                                [512, 512, 'a ceramic mug shaped like a smiling cat', ''],
                            ],
                            advanced=args.advanced)
            with gr.TabItem('Image-to-3D', id='tab_img_to_3d'):
                with gr.Tabs() as sub_tabs_img_to_3d:
                    with gr.TabItem(f'Zero123++ v1.1', id='tab_zero123plus1_1'):
                        _, var_img_to_3d_1_1 = create_interface_img_to_3d(
                            runner.run_segmentation,
                            runner.run_zero123plus,
                            runner.run_zero123plus_to_mesh,
                            api_names=['image_segmentation',
                                       'img_to_3d_1_1_zero123plus',
                                       'img_to_3d_1_1_zero123plus_to_mesh'],
                            init_inverse_steps=640, n_inverse_steps=80, diff_bs=args.diff_bs, advanced=args.advanced)
                    with gr.TabItem(f'Zero123++ v1.2', id='tab_zero123plus1_2'):
                        _, var_img_to_3d_1_2 = create_interface_img_to_3d(
                            runner.run_segmentation,
                            runner.run_zero123plus1_2,
                            runner.run_zero123plus1_2_to_mesh,
                            api_names=[False,
                                       'img_to_3d_1_2_zero123plus',
                                       'img_to_3d_1_2_zero123plus_to_mesh'],
                            init_inverse_steps=720, n_inverse_steps=96, diff_bs=args.diff_bs,
                            advanced=args.advanced, pred_normal=True)
            with gr.TabItem('3D-to-3D', id='tab_3d_to_3d'):
                with gr.Tabs() as sub_tabs_3d_to_3d:
                    with gr.TabItem('Text-Guided', id='tab_text_3d_to_3d'):
                        _, var_text_3d_to_3d = create_interface_3d_to_3d(
                            runner.run_mesh_preproc, runner.run_3d_to_3d,
                            examples=[
                                ['demo/examples_meshes/lara.glb', 3, 'tomb raider lara croft, wearing a backpack', 0.8],
                                ['demo/examples_meshes/zuckerberg_hr.glb', 0, 'elon musk cg rendering', 0.8],
                                ['demo/examples_meshes/camel.glb', -1, 'cg rendering of a camel', 0.8],
                            ],
                            denoising_strength=0.7, api_names=['3d_preproc', 'text_3d_to_3d'],
                            diff_bs=args.diff_bs, advanced=args.advanced)
                    with gr.TabItem('Instruct', id='tab_instruct_3d_to_3d'):
                        _, var_instruct_3d_to_3d = create_interface_3d_to_3d(
                            runner.run_mesh_preproc, runner.run_3d_to_3d,
                            examples=[
                                ['demo/examples_meshes/lara_hr.glb', 3, 'turn her into a cyborg', 1.0],
                                ['demo/examples_meshes/lara_hr.glb', 3, 'as a zelda cosplay, blue outfit', 1.0],
                                ['demo/examples_meshes/cj.glb', 9, 'give him a green jacket', 0.88],
                                ['demo/examples_meshes/polnareff.glb', 9, 'make it a marble roman sculpture', 1.0],
                            ],
                            denoising_strength=1.0, api_names=[False, 'instruct_3d_to_3d'],
                            diff_bs=args.diff_bs, advanced=args.advanced, instruct_3d_to_3d=True)
            with gr.TabItem('Re-Texturing', id='tab_retex'):
                with gr.Tabs() as sub_tabs_retex:
                    with gr.TabItem('Text/Image-Guided', id='tab_text_retex'):
                        _, var_text_retex = create_interface_retexturing(
                            runner.run_mesh_preproc, runner.run_retex,
                            examples=[
                                ['demo/examples_meshes/cj_notex.glb', 9, 'an nba basketball player', '', 0.9, False],
                                ['demo/examples_meshes/robot.glb', 9, 'a robot', '', 0.9, False],
                                ['demo/examples_meshes/napoleon.glb', 9, 'photo of napoleon', 'sculpture', 1.0, False],
                                ['demo/examples_meshes/headphone.glb', -1, 'a close up of a pair of headphones in green cover', '', 0.9, False],
                                ['demo/examples_meshes/car.glb', 6, 'a yellow vintage car', '', 0.9, False],
                            ],
                            denoising_strength=0.9, api_names=[False, 'text_retex'],
                            diff_bs=args.diff_bs, advanced=args.advanced)
                    with gr.TabItem('Instruct', id='tab_instruct_retex'):
                        _, var_instruct_retex = create_interface_retexturing(
                            runner.run_mesh_preproc, runner.run_retex,
                            examples=[
                                ['demo/examples_meshes/cj.glb', 9, 'add dotted printing to his tank top', '', 0.8, True],
                                ['demo/examples_meshes/cj.glb', 9, 'as an nba basketball player', '', 0.8, True],
                            ],
                            denoising_strength=1.0, api_names=[False, 'instruct_retex'],
                            diff_bs=args.diff_bs, advanced=args.advanced, instruct_retex=True)
            with gr.TabItem('Tools', id='tab_tools'):
                with gr.Tabs() as sub_tabs_tools:
                    with gr.TabItem('Export Video', id='tab_export_video'):
                        _, var_3d_to_video = create_interface_3d_to_video(
                            runner.run_mesh_preproc, runner.run_video, api_names=[False, '3d_to_video'])

        for var_dict in [var_stablessdnerf, var_text_3d_to_3d, var_instruct_3d_to_3d, var_img_to_3d_1_1, var_img_to_3d_1_2,
                         var_text_retex, var_instruct_retex]:
            instruct = var_dict.get('instruct', False)
            in_fields = ['output'] if instruct else ['output', 'prompt', 'negative_prompt']
            out_fields = ['in_mesh'] if instruct else ['in_mesh', 'prompt', 'negative_prompt']
            if 'to_text_3d_to_3d' in var_dict:
                var_dict['to_text_3d_to_3d'].click(
                    fn=partial(send_to_click, target_tab_ids=['tab_3d_to_3d', 'tab_text_3d_to_3d']),
                    inputs=[var_dict[k] for k in in_fields],
                    outputs=[var_text_3d_to_3d[k] for k in out_fields] + [main_tabs, sub_tabs_3d_to_3d],
                    show_progress=False,
                    api_name=False
                ).success(
                    **var_text_3d_to_3d['preproc_kwargs'],
                    api_name=False
                )
            if 'to_instruct_3d_to_3d' in var_dict:
                var_dict['to_instruct_3d_to_3d'].click(
                    fn=partial(send_to_click, target_tab_ids=['tab_3d_to_3d', 'tab_instruct_3d_to_3d']),
                    inputs=var_dict['output'],
                    outputs=[var_instruct_3d_to_3d['in_mesh']] + [main_tabs, sub_tabs_3d_to_3d],
                    show_progress=False,
                    api_name=False
                ).success(
                    **var_instruct_3d_to_3d['preproc_kwargs'],
                    api_name=False
                )
            if 'to_text_retex' in var_dict:
                var_dict['to_text_retex'].click(
                    fn=partial(send_to_click, target_tab_ids=['tab_retex', 'tab_text_retex']),
                    inputs=[var_dict[k] for k in in_fields],
                    outputs=[var_text_retex[k] for k in out_fields] + [main_tabs, sub_tabs_retex],
                    show_progress=False,
                    api_name=False
                ).success(
                    **var_text_retex['preproc_kwargs'],
                    api_name=False
                )
            if 'to_instruct_retex' in var_dict:
                var_dict['to_instruct_retex'].click(
                    fn=partial(send_to_click, target_tab_ids=['tab_retex', 'tab_instruct_retex']),
                    inputs=var_dict['output'],
                    outputs=[var_instruct_retex['in_mesh']] + [main_tabs, sub_tabs_retex],
                    show_progress=False,
                    api_name=False
                ).success(
                    **var_instruct_retex['preproc_kwargs'],
                    api_name=False
                )
            if 'export_video' in var_dict:
                var_dict['export_video'].click(
                    fn=partial(send_to_click, target_tab_ids=['tab_tools', 'tab_export_video']),
                    inputs=var_dict['output'],
                    outputs=[var_3d_to_video['in_mesh']] + [main_tabs, sub_tabs_tools],
                    show_progress=False,
                    api_name=False
                ).success(
                    **var_3d_to_video['preproc_kwargs'],
                    api_name=False
                )

        for i, var_img_to_3d in enumerate([var_img_to_3d_1_1, var_img_to_3d_1_2]):
            var_text_to_img_to_3d[f'to_zero123plus1_{i + 1}'].click(
                fn=partial(send_to_click, target_tab_ids=['tab_img_to_3d', f'tab_zero123plus1_{i + 1}']),
                inputs=[var_text_to_img_to_3d[k] for k in ['output_image', 'prompt', 'negative_prompt']],
                outputs=[var_img_to_3d[k] for k in ['in_image', 'prompt', 'negative_prompt']]
                        + [main_tabs, sub_tabs_img_to_3d],
                show_progress=False,
                api_name=False
            ).success(
                fn=lambda: gr.Accordion(open=True),
                inputs=None,
                outputs=var_img_to_3d['prompt_accordion'],
                show_progress=False,
                api_name=False
            )

        demo.queue().launch(
            share=args.share, debug=args.debug
        )


if __name__ == "__main__":
    main()
