import gradio as gr
from functools import partial
from copy import deepcopy
from .gradio_custommodel3d import CustomModel3D
from .shared_opts import create_base_opts, create_generate_bar, create_superres_opts, create_auxiliary_prompt_opts, \
    create_batch_size_opts, create_optimization_opts, set_seed, create_prompt_opts, \
    create_mesh_input, create_send_buttons
from .parameters import retex_defaults, superres_defaults, instruct_retex_params, \
    text_3d_to_3d_superres_params


def create_interface_retexturing(
        preproc_api, mvedit_api, examples=None, denoising_strength=0.7, api_names=None, advanced=True,
        init_inverse_steps=128, n_inverse_steps=48, diff_bs=6, superres_n_inverse_steps=48, instruct_retex=False):
    _retex_defaults = deepcopy(retex_defaults)
    _superres_defaults = deepcopy(superres_defaults)
    _superres_defaults.update(text_3d_to_3d_superres_params)
    if instruct_retex:
        _retex_defaults.update(instruct_retex_params)

    default_var_dict = dict(
        n_inverse_steps=n_inverse_steps, init_inverse_steps=init_inverse_steps, diff_bs=diff_bs)
    default_superres_var_dict = dict(
        n_inverse_steps=superres_n_inverse_steps)

    var_dict = dict(instruct=instruct_retex)
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column():
                create_mesh_input(
                    var_dict, cache_dir=interface.GRADIO_CACHE, preproc_api=preproc_api,
                    render_bs=_retex_defaults['render_bs'],
                    api_name=api_names[0] if api_names is not None else None)
                create_prompt_opts(var_dict)
                if not instruct_retex:
                    with gr.Accordion('Image guidance', open=False):
                        var_dict['in_image'] = gr.Image(type='pil', image_mode='RGBA', label='Input image')
                base_opts = create_base_opts(
                    var_dict, scheduler='EulerAncestralDiscrete',
                    steps=24, denoising_strength=denoising_strength, random_init=False,
                    cfg_scale=_retex_defaults['cfg_scale'], render=False)
                with gr.Column(render=False) as force_auto_uv:
                    var_dict['force_auto_uv'] = gr.Checkbox(
                        label='Force auto UV', value=_retex_defaults['force_auto_uv'], container=False)
                if examples is not None:
                    gr.Examples(
                        examples=examples,
                        inputs=[var_dict[k] for k in
                                ['in_mesh', 'front_view_id', 'prompt', 'negative_prompt', 'denoising_strength', 'force_auto_uv']],
                        outputs=[var_dict['in_mv'], var_dict['proc_mesh']],
                        fn=partial(preproc_api, cache_dir=interface.GRADIO_CACHE,
                                   render_bs=_retex_defaults['render_bs']),
                        run_on_click=True,
                        cache_examples=True,
                        label='Examples (click one of the rows below to start)',
                        api_name=False)
                create_generate_bar(var_dict, text='Generate', seed=-1)
                base_opts.render()
                force_auto_uv.render()
                var_dict['superres'] = dict()
                create_superres_opts(
                    var_dict['superres'], _superres_defaults, do_superres=False, denoising_strength=0.35,
                    n_inverse_steps=superres_n_inverse_steps, show_advanced=advanced)
                if advanced:
                    gr.Markdown('### Advanced settings')
                    var_dict['checkpoint'] = gr.Textbox(
                        label='Stable Diffusion v1.5 checkpoint', lines=1, value=_retex_defaults['checkpoint'],
                        elem_classes=['force-hide-container'])
                    create_auxiliary_prompt_opts(
                        var_dict, **{key: _retex_defaults[key] for key in ['aux_prompt', 'aux_negative_prompt']})
                    create_batch_size_opts(
                        var_dict, diff_bs=diff_bs, patch_bs_nerf=None, **{key: _retex_defaults[key] for key in [
                            'patch_size', 'render_bs', 'patch_bs']})
                    create_optimization_opts(
                        var_dict, n_inverse_steps=n_inverse_steps, init_inverse_steps=init_inverse_steps,
                        tet_resolution=None, tet_init_inverse_steps=None,
                        **{key: _retex_defaults[key] for key in ['start_lr', 'end_lr']})

            with gr.Column():
                var_dict['output'] = CustomModel3D(
                    height=400, label='MVEdit output 3D model', interactive=False)
                create_send_buttons(var_dict)

        default_var_dict = {
            k: default_var_dict.get(k, v) for k, v in _retex_defaults.items()
            if k not in var_dict}
        default_superres_var_dict = {
            'superres_' + k: default_superres_var_dict.get(k, v) for k, v in _superres_defaults.items()
            if k not in var_dict['superres']}
        retex_fun = partial(mvedit_api, **default_var_dict, **default_superres_var_dict,
                            cache_dir=interface.GRADIO_CACHE, instruct=instruct_retex)
        retex_inputs = [var_dict['proc_mesh'], var_dict['front_view_id']] + \
                       [var_dict[k] for k in _retex_defaults.keys()
                        if k not in default_var_dict] + \
                       [var_dict['superres'][k] for k in _superres_defaults.keys()
                        if 'superres_' + k not in default_superres_var_dict]
        if not instruct_retex:
            retex_inputs.append(var_dict['in_image'])

        var_dict['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['seed'],
            outputs=var_dict['last_seed'],
            show_progress=False,
            api_name=False
        ).success(
            fn=retex_fun,
            inputs=[var_dict['last_seed']] + retex_inputs,
            outputs=var_dict['output'],
            concurrency_id='default_group',
            api_name=api_names[1] if api_names is not None else False
        )

        interface.load(**var_dict['preproc_kwargs'], show_api=False)

    return interface, var_dict
