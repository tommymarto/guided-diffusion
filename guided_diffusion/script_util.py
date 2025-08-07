import argparse
import json

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel

NUM_CLASSES = 1000

def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        
        # Distributional loss configuration
        use_distributional=False,
        distributional_track_terms_regardless_of_lambda=False, # can be set to False to save memory and compute, True will track all terms even if lambda is 0
        distributional_kernel="energy",
        distributional_lambda=1.0,
        distributional_lambda_weighting="constant",
        distributional_beta_schedule="constant",
        distributional_population_size=4,
        distributional_kernel_kwargs={"beta": 1.0},
        distributional_loss_weighting="no_weighting",  # can be "no_weighting" or "kingma_snr"
        distributional_num_eps_channels=1,
        dispersion_loss_type="none",
        dispersion_loss_weight=0.5,
        dispersion_loss_last_act_only=False
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        num_classes=NUM_CLASSES,
    )
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    num_classes,
    # Distributional loss configuration
    use_distributional,
    distributional_track_terms_regardless_of_lambda,
    distributional_kernel,
    distributional_lambda,
    distributional_lambda_weighting,
    distributional_beta_schedule,
    distributional_population_size,
    distributional_kernel_kwargs,
    distributional_loss_weighting,
    distributional_num_eps_channels,
    dispersion_loss_type,
    dispersion_loss_weight,
    dispersion_loss_last_act_only
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        num_classes=num_classes,
        use_distributional=use_distributional,
        distributional_num_eps_channels=distributional_num_eps_channels,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_distributional=use_distributional,
        distributional_track_terms_regardless_of_lambda=distributional_track_terms_regardless_of_lambda,
        distributional_kernel=distributional_kernel,
        distributional_lambda=distributional_lambda,
        distributional_lambda_weighting=distributional_lambda_weighting,
        distributional_beta_schedule=distributional_beta_schedule,
        distributional_population_size=distributional_population_size,
        distributional_kernel_kwargs=distributional_kernel_kwargs,
        distributional_loss_weighting=distributional_loss_weighting,
        distributional_num_eps_channels=distributional_num_eps_channels,
        dispersion_loss_type=dispersion_loss_type,
        dispersion_loss_weight=dispersion_loss_weight,
        dispersion_loss_last_act_only=dispersion_loss_last_act_only
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    num_classes=NUM_CLASSES,
    use_distributional=False,
    distributional_num_eps_channels=1,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        use_distributional=use_distributional,
        distributional_num_eps_channels=distributional_num_eps_channels,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    use_distributional=False,
    distributional_track_terms_regardless_of_lambda=False,
    distributional_kernel="energy",
    distributional_lambda=1.0,
    distributional_lambda_weighting="constant",
    distributional_beta_schedule="constant",
    distributional_population_size=4,
    distributional_kernel_kwargs={"beta": 1.0},
    distributional_loss_weighting="no_weighting",
    distributional_num_eps_channels=1,
    dispersion_loss_type="none",
    dispersion_loss_weight=0.5,
    dispersion_loss_last_act_only=False
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    elif use_distributional:
        loss_type = gd.LossType.DISTRIBUTIONAL
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        distributional_track_terms_regardless_of_lambda=distributional_track_terms_regardless_of_lambda,
        distributional_kernel=distributional_kernel,
        distributional_lambda=distributional_lambda,
        distributional_lambda_weighting=distributional_lambda_weighting,
        distributional_beta_schedule=distributional_beta_schedule,
        distributional_population_size=distributional_population_size,
        distributional_kernel_kwargs=distributional_kernel_kwargs,
        distributional_loss_weighting=gd.LossWeighting(distributional_loss_weighting.upper()),
        distributional_num_eps_channels=distributional_num_eps_channels,
        dispersion_loss_type=gd.DispersionLossType[dispersion_loss_type.upper()],
        dispersion_loss_weight=dispersion_loss_weight,
        dispersion_loss_last_act_only=dispersion_loss_last_act_only
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        elif isinstance(v, dict):
            v_type = json.loads
        elif isinstance(v, list):
            if len(v) > 0 and type(v[0]) is not None:
                v_type = type(v[0])
            else:
                v_type = str
            parser.add_argument(f"--{k}", default=v, type=v_type, nargs='+' if len(v) > 1 else '*')
            continue
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
