import gradio as gr
from functools import partial
from .gradio_custommodel3d import CustomModel3D
from .shared_opts import create_base_opts, create_generate_bar, create_superres_opts, create_auxiliary_prompt_opts, \
    create_base_advanced_opts, create_loss_sliders, create_optimization_opts, set_seed, create_prompt_opts, \
    create_mesh_send_buttons
from .parameters import nerf_mesh_defaults, superres_defaults


def create_passes(output_list, num_passes, height=172, image_mode='RGBA'):
    with gr.Tabs():
        for pass_id in range(6):
            with gr.TabItem(f'Pass {pass_id}'):
                with gr.Row():
                    for view_id in range(num_passes):
                        output_list.append(gr.Image(
                            interactive=False, label='Zero123++ output', show_label=view_id == 0,
                            height=height, min_width=140, image_mode=image_mode))


def create_interface_img_to_3d(
        segmentation_api, zero123plus_api, mvedit_api, api_names=None, advanced=True, init_inverse_steps=640,
        n_inverse_steps=96, diff_bs=6, tet_resolution=128, superres_n_inverse_steps=640, num_passes=6,
        pred_normal=False):

    default_var_dict = dict(
        init_inverse_steps=init_inverse_steps, n_inverse_steps=n_inverse_steps, diff_bs=diff_bs,
        tet_resolution=tet_resolution)
    default_superres_var_dict = dict(
        n_inverse_steps=superres_n_inverse_steps)

    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    var_dict['in_image'] = gr.Image(
                        type='pil', image_mode='RGBA', label='Input image')
                    var_dict['fg_image'] = gr.Image(
                        type='pil', label='Segmented foreground', interactive=False, image_mode='RGBA')
                gr.Examples(
                    examples='demo/examples_images',
                    inputs=var_dict['in_image'],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=20)
                with gr.Accordion(
                        'Optional text prompts', open=True, elem_classes=['custom-spacing']) as prompt_accordion:
                    create_prompt_opts(var_dict)
                var_dict['prompt_accordion'] = prompt_accordion
                var_dict['gen_all'] = dict()
                create_generate_bar(var_dict['gen_all'], text='Generate', seed=-1)
                create_base_opts(
                    var_dict, scheduler='DPMSolverMultistep',
                    steps=24, denoising_strength=0.5, random_init=False,
                    cfg_scale=nerf_mesh_defaults['cfg_scale'])
                var_dict['superres'] = dict()
                create_superres_opts(
                    var_dict['superres'], superres_defaults,
                    use_ip_adapter=True, scheduler='DPMSolverSDEKarras',
                    n_inverse_steps=superres_n_inverse_steps, show_advanced=advanced)
                if advanced:
                    gr.Markdown('### Advanced settings')
                    var_dict['checkpoint'] = gr.Textbox(
                        label='Stable Diffusion v1.5 checkpoint', lines=1, value=nerf_mesh_defaults['checkpoint'],
                        elem_classes=['force-hide-container'])
                    create_auxiliary_prompt_opts(
                        var_dict, **{key: nerf_mesh_defaults[key] for key in ['aux_prompt', 'aux_negative_prompt']})
                    create_base_advanced_opts(
                        var_dict, diff_bs=diff_bs, **{key: nerf_mesh_defaults[key] for key in [
                            'patch_size', 'patch_bs_nerf', 'render_bs', 'patch_bs',
                            'max_num_views', 'min_num_views', 'mvedit_mode']})
                    create_loss_sliders(
                        var_dict, **{key: nerf_mesh_defaults[key] for key in [
                            'alpha_soften', 'normal_reg_weight', 'start_entropy_weight', 'end_entropy_weight',
                            'entropy_d', 'mesh_smoothness']})
                    create_optimization_opts(
                        var_dict, n_inverse_steps=n_inverse_steps, init_inverse_steps=init_inverse_steps,
                        tet_resolution=tet_resolution,
                        **{key: nerf_mesh_defaults[key] for key in [
                            'tet_init_inverse_steps', 'start_lr', 'end_lr', 'ingp_resolution']})

            with gr.Column(scale=2):
                var_dict['gen_zero123plus'] = dict()
                create_generate_bar(
                    var_dict['gen_zero123plus'], text='Run Zero123++ only', variant='secondary', seed=-1)
                var_dict['zero123plus_outputs'] = []
                if pred_normal:
                    with gr.Tabs():
                        with gr.TabItem('RGB'):
                            create_passes(var_dict['zero123plus_outputs'], num_passes, height=158)
                        with gr.TabItem('Normal'):
                            create_passes(var_dict['zero123plus_outputs'], num_passes, height=158, image_mode='RGB')
                else:
                    create_passes(var_dict['zero123plus_outputs'], num_passes)
                var_dict['gen_mvedit'] = dict()
                create_generate_bar(
                    var_dict['gen_mvedit'], text='Run MVEdit only', variant='secondary', seed=-1)
                var_dict['output'] = CustomModel3D(
                    height=400, label='MVEdit output 3D model', camera_position=(180, 80, 3.0), interactive=False)
                create_mesh_send_buttons(var_dict)

        default_var_dict = {
            k: default_var_dict.get(k, v) for k, v in nerf_mesh_defaults.items()
            if k not in var_dict}
        default_superres_var_dict = {
            'superres_' + k: default_superres_var_dict.get(k, v) for k, v in superres_defaults.items()
            if k not in var_dict['superres']}
        img_to_3d_fun = partial(mvedit_api, **default_var_dict, **default_superres_var_dict,
                                cache_dir=interface.GRADIO_CACHE)
        img_to_3d_inputs = [var_dict['fg_image']] + \
                           [var_dict[k] for k in nerf_mesh_defaults.keys()
                            if k not in default_var_dict] + \
                           [var_dict['superres'][k] for k in superres_defaults.keys()
                            if 'superres_' + k not in default_superres_var_dict] + \
                           var_dict['zero123plus_outputs']

        var_dict['gen_all']['run_btn'].click(
            fn=segmentation_api, inputs=var_dict['in_image'],
            outputs=var_dict['fg_image'], concurrency_id='default_group',
            api_name=api_names[0] if api_names is not None else False
        ).success(
            fn=set_seed,
            inputs=var_dict['gen_all']['seed'],
            outputs=var_dict['gen_all']['last_seed'],
            show_progress=False,
            api_name=False
        ).success(
            fn=zero123plus_api,
            inputs=[var_dict['gen_all']['last_seed'], var_dict['fg_image']],
            outputs=var_dict['zero123plus_outputs'], concurrency_id='default_group',
            api_name=api_names[1] if api_names is not None else False
        ).success(
            fn=img_to_3d_fun,
            inputs=[var_dict['gen_all']['last_seed']] + img_to_3d_inputs,
            outputs=var_dict['output'],
            concurrency_id='default_group',
            api_name=api_names[2] if api_names is not None else False
        )

        var_dict['gen_zero123plus']['run_btn'].click(
            fn=segmentation_api, inputs=var_dict['in_image'],
            outputs=var_dict['fg_image'], api_name=False, concurrency_id='default_group'
        ).success(
            fn=set_seed,
            inputs=var_dict['gen_zero123plus']['seed'],
            outputs=var_dict['gen_zero123plus']['last_seed'],
            show_progress=False,
            api_name=False
        ).success(
            fn=zero123plus_api,
            inputs=[var_dict['gen_zero123plus']['last_seed'], var_dict['fg_image']],
            outputs=var_dict['zero123plus_outputs'], api_name=False, concurrency_id='default_group'
        )

        var_dict['gen_mvedit']['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['gen_mvedit']['seed'],
            outputs=var_dict['gen_mvedit']['last_seed'],
            show_progress=False,
            api_name=False
        ).success(
            fn=img_to_3d_fun,
            inputs=[var_dict['gen_mvedit']['last_seed']] + img_to_3d_inputs,
            outputs=var_dict['output'],
            api_name=False, concurrency_id='default_group'
        )

    return interface, var_dict
