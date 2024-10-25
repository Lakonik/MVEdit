import gradio as gr
from functools import partial
from copy import deepcopy
from .gradio_custommodel3d import CustomModel3D
from .shared_opts import create_base_opts, create_generate_bar, create_superres_opts, create_auxiliary_prompt_opts, \
    create_base_advanced_opts, create_loss_sliders, create_optimization_opts, set_seed, create_prompt_opts, \
    create_mesh_input, create_mesh_send_buttons
from .parameters import nerf_mesh_defaults, superres_defaults, \
    text_3d_to_3d_params, text_3d_to_3d_superres_params, instruct_3d_to_3d_params


def create_interface_3d_to_3d(
        preproc_api, mvedit_api, examples=None, denoising_strength=0.7, api_names=None, advanced=True,
        init_inverse_steps=640, n_inverse_steps=96, diff_bs=6, tet_resolution=256,
        superres_n_inverse_steps=640, instruct_3d_to_3d=False):
    _nerf_mesh_defaults = deepcopy(nerf_mesh_defaults)
    _superres_defaults = deepcopy(superres_defaults)
    _nerf_mesh_defaults.update(instruct_3d_to_3d_params if instruct_3d_to_3d else text_3d_to_3d_params)
    _superres_defaults.update(text_3d_to_3d_superres_params)

    default_var_dict = dict(
        init_inverse_steps=init_inverse_steps, n_inverse_steps=n_inverse_steps, diff_bs=diff_bs,
        tet_resolution=tet_resolution)
    default_superres_var_dict = dict(
        n_inverse_steps=superres_n_inverse_steps)

    var_dict = dict(instruct=instruct_3d_to_3d)
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column():
                create_mesh_input(
                    var_dict, cache_dir=interface.GRADIO_CACHE, preproc_api=preproc_api,
                    render_bs=_nerf_mesh_defaults['render_bs'],
                    api_name=api_names[0] if api_names is not None else None)
                create_prompt_opts(var_dict)
                base_opts = create_base_opts(
                    var_dict,
                    steps=32, denoising_strength=denoising_strength, random_init=False,
                    cfg_scale=_nerf_mesh_defaults['cfg_scale'], render=False)
                if examples is not None:
                    gr.Examples(
                        examples=examples,
                        inputs=[var_dict[k] for k in ['in_mesh', 'front_view_id', 'prompt', 'denoising_strength', 'cfg_scale']],
                        outputs=[var_dict['in_mv'], var_dict['proc_mesh']],
                        fn=partial(preproc_api, cache_dir=interface.GRADIO_CACHE,
                                   render_bs=_nerf_mesh_defaults['render_bs']),
                        run_on_click=True,
                        cache_examples=True,
                        label='Examples (click one of the rows below to start)',
                        api_name=False)
                create_generate_bar(var_dict, text='Generate', seed=-1)
                base_opts.render()
                var_dict['superres'] = dict()
                create_superres_opts(
                    var_dict['superres'], _superres_defaults, denoising_strength=0.4,
                    n_inverse_steps=superres_n_inverse_steps, show_advanced=advanced)
                if advanced:
                    gr.Markdown('### Advanced settings')
                    var_dict['checkpoint'] = gr.Textbox(
                        label='Stable Diffusion v1.5 checkpoint', lines=1, value=_nerf_mesh_defaults['checkpoint'],
                        elem_classes=['force-hide-container'])
                    create_auxiliary_prompt_opts(
                        var_dict, **{key: _nerf_mesh_defaults[key] for key in ['aux_prompt', 'aux_negative_prompt']})
                    create_base_advanced_opts(
                        var_dict, diff_bs=diff_bs, **{key: _nerf_mesh_defaults[key] for key in [
                            'patch_size', 'patch_bs_nerf', 'render_bs', 'patch_bs',
                            'max_num_views', 'min_num_views', 'mvedit_mode']})
                    create_loss_sliders(
                        var_dict, **{key: _nerf_mesh_defaults[key] for key in [
                            'alpha_soften', 'normal_reg_weight', 'start_entropy_weight', 'end_entropy_weight',
                            'entropy_d', 'mesh_smoothness']})
                    create_optimization_opts(
                        var_dict, n_inverse_steps=n_inverse_steps, init_inverse_steps=init_inverse_steps,
                        tet_resolution=tet_resolution,
                        **{key: _nerf_mesh_defaults[key] for key in [
                            'tet_init_inverse_steps', 'start_lr', 'end_lr', 'ingp_resolution']})

            with gr.Column():
                var_dict['output'] = CustomModel3D(
                    height=400, label='Output 3D model', interactive=False)
                create_mesh_send_buttons(var_dict)

        default_var_dict = {
            k: default_var_dict.get(k, v) for k, v in _nerf_mesh_defaults.items()
            if k not in var_dict}
        default_superres_var_dict = {
            'superres_' + k: default_superres_var_dict.get(k, v) for k, v in _superres_defaults.items()
            if k not in var_dict['superres']}
        text_3d_to_3d_fun = partial(mvedit_api, **default_var_dict, **default_superres_var_dict,
                                    cache_dir=interface.GRADIO_CACHE, instruct=instruct_3d_to_3d)
        text_3d_to_3d_inputs = [var_dict['proc_mesh'], var_dict['front_view_id']] + \
                               [var_dict[k] for k in _nerf_mesh_defaults.keys()
                                if k not in default_var_dict] + \
                               [var_dict['superres'][k] for k in _superres_defaults.keys()
                                if 'superres_' + k not in default_superres_var_dict]

        var_dict['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['seed'],
            outputs=var_dict['last_seed'],
            show_progress=False,
            api_name=False
        ).success(
            fn=text_3d_to_3d_fun,
            inputs=[var_dict['last_seed']] + text_3d_to_3d_inputs,
            outputs=var_dict['output'],
            concurrency_id='default_group',
            api_name=api_names[1] if api_names is not None else False
        )

        interface.load(**var_dict['preproc_kwargs'], show_api=False)

    return interface, var_dict
