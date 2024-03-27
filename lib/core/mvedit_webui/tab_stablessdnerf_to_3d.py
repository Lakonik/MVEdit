import gradio as gr
from functools import partial
from copy import deepcopy
from .gradio_custommodel3d import CustomModel3D
from .shared_opts import create_base_opts, create_generate_bar, create_superres_opts, create_auxiliary_prompt_opts, \
    create_batch_size_opts, create_loss_sliders, create_optimization_opts, set_seed, create_prompt_opts, \
    create_stablessdnerf_opts, create_send_buttons
from .parameters import nerf_mesh_defaults, superres_defaults, \
    stablessdnerf_to_mesh_params, text_3d_to_3d_superres_params, stablessdnerf_signatures


def create_interface_stablessdnerf_to_3d(
        stablessdnerf_api, mvedit_api, examples=None, denoising_strength=0.7, api_names=None, advanced=True,
        init_inverse_steps=0, n_inverse_steps=96, diff_bs=6, tet_resolution=256,
        superres_n_inverse_steps=48):
    _nerf_mesh_defaults = deepcopy(nerf_mesh_defaults)
    _superres_defaults = deepcopy(superres_defaults)
    _nerf_mesh_defaults.update(stablessdnerf_to_mesh_params)
    _superres_defaults.update(text_3d_to_3d_superres_params)

    default_stablessdnerf_var_dict = dict()
    default_var_dict = dict(
        init_inverse_steps=init_inverse_steps, n_inverse_steps=n_inverse_steps, diff_bs=diff_bs,
        tet_resolution=tet_resolution)
    default_superres_var_dict = dict(
        n_inverse_steps=superres_n_inverse_steps)

    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        md_txt = ('Tip: It\'s highly recommended to send the results **to Re-texturing** for texture refinement '
                  '(in this case Texture super-resolution can be unticked).')
        gr.Markdown(md_txt)
        with gr.Row():
            with gr.Column():
                create_prompt_opts(var_dict)
                base_opts = create_base_opts(
                    var_dict, scheduler='DPMSolverMultistep',
                    steps=32, denoising_strength=denoising_strength, random_init=False,
                    cfg_scale=_nerf_mesh_defaults['cfg_scale'], render=False)
                if examples is not None:
                    gr.Examples(
                        examples=examples,
                        inputs=[var_dict[k] for k in ['prompt', 'negative_prompt']],
                        label='Examples (click one of the rows below to start)',
                        api_name=False)
                create_generate_bar(var_dict, text='Generate', seed=-1)
                var_dict['stablessdnerf'] = dict()
                create_stablessdnerf_opts(var_dict['stablessdnerf'], stablessdnerf_signatures)
                var_dict['stablessdnerf'].update(prompt=var_dict['prompt'], negative_prompt=var_dict['negative_prompt'])
                base_opts.render()
                var_dict['superres'] = dict()
                create_superres_opts(
                    var_dict['superres'], _superres_defaults, denoising_strength=0.5,
                    n_inverse_steps=superres_n_inverse_steps, show_advanced=advanced)
                if advanced:
                    gr.Markdown('### Advanced settings')
                    var_dict['checkpoint'] = gr.Textbox(
                        label='Stable Diffusion v1.5 checkpoint', lines=1, value=_nerf_mesh_defaults['checkpoint'],
                        elem_classes=['force-hide-container'])
                    create_auxiliary_prompt_opts(
                        var_dict, **{key: _nerf_mesh_defaults[key] for key in ['aux_prompt', 'aux_negative_prompt']})
                    create_batch_size_opts(
                        var_dict, diff_bs=diff_bs, **{key: _nerf_mesh_defaults[key] for key in [
                            'patch_size', 'patch_bs_nerf', 'render_bs', 'patch_bs']})
                    create_loss_sliders(
                        var_dict, **{key: _nerf_mesh_defaults[key] for key in [
                            'alpha_soften', 'normal_reg_weight', 'start_entropy_weight', 'end_entropy_weight',
                            'entropy_d', 'mesh_smoothness']})
                    create_optimization_opts(
                        var_dict, n_inverse_steps=n_inverse_steps, init_inverse_steps=init_inverse_steps,
                        tet_resolution=tet_resolution,
                        **{key: _nerf_mesh_defaults[key] for key in ['tet_init_inverse_steps', 'start_lr', 'end_lr']})

            with gr.Column():
                var_dict['gen_stablessdnerf'] = dict()
                create_generate_bar(
                    var_dict['gen_stablessdnerf'], text='Run StableSSDNeRF only', variant='secondary', seed=-1)
                var_dict['output_video'] = gr.Video(
                    label='StableSSDNeRF output', interactive=False, width=250, height=250, autoplay=True)
                with gr.Column(visible=False):
                    var_dict['output_triplane'] = gr.Textbox(label='StableSSDNeRF output triplane')
                var_dict['gen_mvedit'] = dict()
                create_generate_bar(
                    var_dict['gen_mvedit'], text='Run MVEdit only', variant='secondary', seed=-1)
                var_dict['output'] = CustomModel3D(
                    height=400, label='MVEdit output 3D model', interactive=False)
                create_send_buttons(var_dict)

        default_stablessdnerf_var_dict = {
            k: default_stablessdnerf_var_dict.get(k, v) for k, v in stablessdnerf_signatures.items()
            if k not in var_dict['stablessdnerf']}
        default_var_dict = {
            k: default_var_dict.get(k, v) for k, v in _nerf_mesh_defaults.items()
            if k not in var_dict}
        default_superres_var_dict = {
            'superres_' + k: default_superres_var_dict.get(k, v) for k, v in _superres_defaults.items()
            if k not in var_dict['superres']}

        stablessdnerf_fun = partial(stablessdnerf_api, **default_stablessdnerf_var_dict,
                                    cache_dir=interface.GRADIO_CACHE)
        stablessdnerf_inputs = [var_dict['stablessdnerf'][k] for k in stablessdnerf_signatures.keys()
                                if k not in default_stablessdnerf_var_dict]

        text_3d_to_3d_fun = partial(mvedit_api, **default_var_dict, **default_superres_var_dict,
                                    cache_dir=interface.GRADIO_CACHE)
        text_3d_to_3d_inputs = [var_dict['output_triplane']] + \
                               [var_dict[k] for k in _nerf_mesh_defaults.keys()
                                if k not in default_var_dict] + \
                               [var_dict['superres'][k] for k in _superres_defaults.keys()
                                if 'superres_' + k not in default_superres_var_dict]

        var_dict['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['seed'],
            outputs=var_dict['last_seed'], api_name=False
        ).success(
            fn=stablessdnerf_fun,
            inputs=[var_dict['last_seed']] + stablessdnerf_inputs,
            outputs=[var_dict['output_video'], var_dict['output_triplane']],
            concurrency_id='default_group',
            api_name=api_names[0] if api_names is not None else False
        ).success(
            fn=text_3d_to_3d_fun,
            inputs=[var_dict['last_seed']] + text_3d_to_3d_inputs,
            outputs=var_dict['output'],
            concurrency_id='default_group',
            api_name=api_names[1] if api_names is not None else False
        )

        var_dict['gen_stablessdnerf']['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['gen_stablessdnerf']['seed'],
            outputs=var_dict['gen_stablessdnerf']['last_seed'], api_name=False
        ).success(
            fn=stablessdnerf_fun,
            inputs=[var_dict['gen_stablessdnerf']['last_seed']] + stablessdnerf_inputs,
            outputs=[var_dict['output_video'], var_dict['output_triplane']],
            concurrency_id='default_group',
            api_name=False
        )

        var_dict['gen_mvedit']['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['gen_mvedit']['seed'],
            outputs=var_dict['gen_mvedit']['last_seed'], api_name=False
        ).success(
            fn=text_3d_to_3d_fun,
            inputs=[var_dict['gen_mvedit']['last_seed']] + text_3d_to_3d_inputs,
            outputs=var_dict['output'],
            concurrency_id='default_group',
            api_name=False
        )

    return interface, var_dict
