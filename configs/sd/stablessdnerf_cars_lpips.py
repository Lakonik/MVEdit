# Note: please prepare the initial latents (cache/stablessdnerf_cars/code) before training this model.
name = 'stablessdnerf_cars_lpips'

model = dict(
    type='DiffusionNeRFText',
    code_size=(3, 4, 40, 40),
    code_permute=(1, 0, 2, 3),
    code_reshape=(4, 120, 40),
    code_activation=dict(
        type='NormalizedTanhCode', mean=0.0, std=0.5, clip_range=3),
    grid_size=32,
    diffusion=dict(
        type='GaussianDiffusionText',
        num_timesteps=1000,
        betas_cfg=dict(type='linear'),
        text_encoder=dict(
            type='CLIPLoRAWrapper',
            lora_layers=dict(
                type='LoRAAttnProcessor2_0',
                rank=4),
            text_encoder=dict(
                type='CLIPTextModel',
                freeze=True,
                pretrained='huggingface://stabilityai/stable-diffusion-2/text_encoder/pytorch_model.bin',
                max_position_embeddings=77)),
        tokenizer='stabilityai/stable-diffusion-2',
        denoising=dict(
            type='UNetLoRAWrapper',
            lora_layers=dict(
                type='LoRAAttnProcessor2_0',
                rank=32),
            unet=dict(
                type='UNet2DConditionModel',
                freeze=True,
                pretrained='huggingface://stabilityai/stable-diffusion-2/unet/diffusion_pytorch_model.bin',
                act_fn='silu',
                attention_head_dim=[5, 10, 20, 20],
                block_out_channels=[320, 640, 1280, 1280],
                center_input_sample=False,
                cross_attention_dim=1024,
                down_block_types=[
                    'CrossAttnDownBlock2D',
                    'CrossAttnDownBlock2D',
                    'CrossAttnDownBlock2D',
                    'DownBlock2D'],
                downsample_padding=1,
                dual_cross_attention=False,
                flip_sin_to_cos=True,
                freq_shift=0,
                in_channels=4,
                layers_per_block=2,
                mid_block_scale_factor=1,
                norm_eps=1e-5,
                norm_num_groups=32,
                out_channels=4,
                sample_size=96,
                up_block_types=[
                    'UpBlock2D',
                    'CrossAttnUpBlock2D',
                    'CrossAttnUpBlock2D',
                    'CrossAttnUpBlock2D'],
                use_linear_projection=True)),
        timestep_sampler=dict(
            type='SNRWeightedTimeStepSampler',
            power=0.5),
        ddpm_loss=dict(
            type='DDPMMSELossMod',
            rescale_mode='timestep_weight',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=1000),
            data_info=dict(pred='v_t_pred', target='v_t'),
            weight_scale=20)),
    decoder=dict(
        type='TriPlaneDecoder',
        preprocessor=dict(
            type='VAEDecoder',
            in_channels=12,
            out_channels=48,
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'),
            block_out_channels=(128, 256),
            layers_per_block=2),
        code_reshape=(12, 40, 40),
        preproc_size=(3, 16, 80, 80),
        plane_cfg=['yx', 'yz', 'xz'],
        interp_mode='bilinear',
        flip_z=True,
        base_layers=[48, 64],
        density_layers=[64, 1],
        color_layers=[64, 3],
        use_dir_enc=True,
        dir_layers=[16, 64],
        activation='silu',
        sigma_activation='trunc_exp',
        sigmoid_saturation=0.001,
        max_steps=256),
    decoder_use_ema=True,
    freeze_decoder=False,
    bg_color=1,
    pixel_loss=dict(
        type='L1LossMod',
        loss_weight=1.2),
    patch_loss=dict(
        type='LPIPSLoss',
        loss_weight=1.2),
    cache_size=2458,
    cache_16bit=True)

save_interval = 5000
eval_interval = 10000
code_dir = 'cache/' + name + '/code'
work_dir = 'work_dirs/' + name

train_cfg = dict(
    uncond_prob=0.1,
    dt_gamma_scale=0.5,
    density_thresh=0.1,
    extra_scene_step=0,
    n_inverse_rays=2 ** 12,
    n_decoder_rays=2 ** 12,
    loss_coef=0.1 / (128 * 128),
    optimizer=dict(type='Adam', lr=0.01, weight_decay=0.),
    freeze_code=True,
    decoder_grad_clip=0.1,
    cache_load_from=code_dir,
    viz_dir=None)
test_cfg = dict(
    img_size=(128, 128),
    num_timesteps=50,
    clip_range=[-3, 3],
    density_thresh=0.1,
    density_step=8,
    cfg_scale=1.0)

optimizer = {
    'diffusion.text_encoder.lora_layers': dict(type='AdamW', lr=2e-4),
    'diffusion.denoising': dict(
        type='AdamW',
        lr=1e-5,
        paramwise_cfg=dict(
            custom_keys={
                'lora': dict(lr_mult=20.0)
            })),
    'decoder': dict(type='AdamW', lr=1e-3)}
dataset_type = 'ShapeNetSRN'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        caption_path='data/shapenet/captions.pkl',
        data_prefix='data/shapenet/cars_train',
        cache_path='data/shapenet/cars_caption_train_cache.pkl'),
    val_text=dict(
        type=dataset_type,
        caption_path='data/shapenet/captions.pkl',
        data_prefix='data/shapenet/cars_test',
        load_imgs=False,
        num_test_imgs=251,
        scene_id_as_name=True,
        cache_path='data/shapenet/cars_caption_test_cache.pkl'),
    train_dataloader=dict(split_data=True))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    gamma=0.5,
    step=[35000])
checkpoint_config = dict(interval=save_interval, by_epoch=False)

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_text',
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=32,
        metrics=dict(
            type='FIDKID',
            num_images=704 * 251,
            inception_pkl='work_dirs/cache/cars_test_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),
        viz_dir=work_dir + '/viz_text',
        save_best_ckpt=False)]

total_iters = 40000  # not improving after this
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHookMod',
        module_keys=('decoder_ema', 'diffusion_ema'),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=4, ema_rampup=0.05, batch_size=16, eps=1e-8),
        priority='VERY_HIGH'),
    dict(
        type='SaveCacheHook',
        interval=save_interval,
        by_epoch=False,
        out_dir=code_dir,
        viz_dir='cache/' + name + '/viz'),
    dict(
        type='ModelUpdaterHook',
        step=[2000, 5000, 35000],
        cfgs=[{'train_cfg.n_inverse_rays': 2 ** 14,
               'train_cfg.n_decoder_rays': 2 ** 14},
              {'train_cfg.freeze_code': False},
              {'train_cfg.optimizer.lr': 0.005}],
        by_epoch=False)
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunnerMod',
    is_dynamic_ddp=False,
    pass_training_status=True,
    ckpt_trainable_only=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
