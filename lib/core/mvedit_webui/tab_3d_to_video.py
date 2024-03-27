import gradio as gr
from functools import partial
from .shared_opts import create_mesh_input
from .parameters import nerf_mesh_defaults


def create_interface_3d_to_video(preproc_api, mvedit_api, api_names=None):

    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column():
                create_mesh_input(
                    var_dict, cache_dir=interface.GRADIO_CACHE, preproc_api=preproc_api,
                    render_bs=nerf_mesh_defaults['render_bs'],
                    api_name=api_names[0] if api_names is not None else None)
                var_dict['render'] = gr.Button('Render', variant='primary')
                var_dict['lossless'] = gr.Checkbox(
                    label='Lossless (incompatible with some players)', value=False, container=False)
                with gr.Column(variant='compact', elem_classes=['custom-spacing']):
                    with gr.Row(variant='compact', elem_classes=['force-hide-container']):
                        var_dict['layer'] = gr.Dropdown(
                            ['RGB', 'Normal'], value='RGB', label='Layer',
                            elem_classes=['force-hide-container'])
                        var_dict['elevation'] = gr.Slider(
                            label='Elevation', minimum=-179, maximum=179, step=1.0, value=10.0,
                            elem_classes=['force-hide-container'])
                    with gr.Row(variant='compact', elem_classes=['force-hide-container']):
                        var_dict['distance'] = gr.Slider(
                            label='Distance', minimum=1.0, maximum=10.0, step=0.1, value=4.0,
                            elem_classes=['force-hide-container'])
                        var_dict['fov'] = gr.Slider(
                            label='FoV', minimum=10.0, maximum=70.0, step=1.0, value=30,
                            elem_classes=['force-hide-container'])
                    with gr.Row(variant='compact', elem_classes=['force-hide-container']):
                        var_dict['length'] = gr.Slider(
                            label='Length (sec)', minimum=1, maximum=20, step=0.5, value=10,
                            elem_classes=['force-hide-container'])
                        var_dict['resolution'] = gr.Slider(
                            label='Resolution', minimum=64, maximum=1024, step=8, value=512,
                            elem_classes=['force-hide-container'])

            with gr.Column():
                var_dict['output_video'] = gr.Video(
                    label='Output video', interactive=False, autoplay=True)

        video_fun = partial(
            mvedit_api, cache_dir=interface.GRADIO_CACHE, render_bs=nerf_mesh_defaults['render_bs'])

        var_dict['render'].click(
            fn=video_fun,
            inputs=[var_dict['proc_mesh'], var_dict['front_view_id'],
                    var_dict['distance'], var_dict['elevation'], var_dict['fov'],
                    var_dict['length'], var_dict['resolution'], var_dict['lossless'], var_dict['layer']],
            outputs=var_dict['output_video'],
            concurrency_id='default_group',
            api_name=api_names[1] if api_names is not None else False
        )

        interface.load(**var_dict['preproc_kwargs'], show_api=False)

    return interface, var_dict
