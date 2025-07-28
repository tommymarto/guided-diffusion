#%%
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os

from einops import rearrange, repeat
import numpy as np
import torch as th
# import torch.distributed as dist
from PIL import Image
from torchvision.utils import make_grid
from guided_diffusion import logger
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    seed_everything,
)

import matplotlib.pyplot as plt
from io import BytesIO


class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

@th.no_grad()
def display_image_grid(
    tensor,
    save_path=None,
    factor=1,
    display=False,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", th.uint8).numpy()
    im = Image.fromarray(ndarr)
    w, h = im.size
    if factor > 1:
        im = im.resize((w * factor, h * factor), Image.NEAREST)
        # im = im.resize((w * factor, h * factor), Image.LANCZOS)
    if save_path is not None:
        im.save(save_path)
    if display:
        from IPython.display import display as ipy_display
        ipy_display(im)
    return im

#%%

def timestamp_sample(
    diffusion: GaussianDiffusion,
    model,
    imgs,
    model_kwargs=None,
    use_ddim=False,
    num_timesteps=50,
    device=None,
    factor=1,
    exp_name=None,
):
    """
    Generate samples from the model and yield intermediate samples from
    each timestep of diffusion.

    Arguments are the same as p_sample_loop().
    Returns a generator over dicts, where each dict is the return value of
    p_sample().
    """
    t = th.linspace(
        diffusion.num_timesteps - 1, 0, num_timesteps, device=device
    ).long()
    noise = th.randn_like(imgs)

    t_forward = repeat(t, 't -> b t', b=imgs.shape[0])
    imgs_forward = repeat(imgs, 'b ... -> b t ...', t=t_forward.shape[1])
    noise_forward = repeat(noise, 'b ... -> b t ...', t=t_forward.shape[1])
    
    if "y" in model_kwargs:
        model_kwargs["y"] = repeat(model_kwargs["y"], 'b -> (b t)', t=t_forward.shape[1]).to(device)
    if diffusion.use_distributional:
        eps = th.randn_like(imgs)
        model_kwargs["eps"] = repeat(eps, 'b ... -> (b t) ...', t=t_forward.shape[1]).to(device)

    x_t_forward = diffusion.q_sample(imgs_forward, t_forward, noise=noise_forward)

    sample_fn = (
        diffusion.p_sample if not use_ddim else diffusion.ddim_sample
    )
    
    out = sample_fn(
        model,
        rearrange(x_t_forward, 'b t ... -> (b t) ...'),
        rearrange(t_forward, 'b t -> (b t)'),
        clip_denoised=False,
        model_kwargs=model_kwargs,
    )
    
    
    pred_xstart = rearrange(out["pred_xstart"], '(b t) ... -> b t ...', b=imgs.shape[0])
    
    # Interleave original images and predicted x_start
    # Stack them along a new dimension, then flatten
    interleaved = th.stack([x_t_forward, pred_xstart], dim=2)
    interleaved = rearrange(interleaved, 'b t i c h w -> (b i t) c h w')

    
    image_grid_pil = display_image_grid(interleaved, nrow=num_timesteps, normalize=True, value_range=(-1, 1), factor=factor)
    
    
    # Create noise level plot
    noise_levels = diffusion.sqrt_one_minus_alphas_cumprod[t.cpu().numpy()]
    fig, ax = plt.subplots(figsize=(image_grid_pil.width / 100, 1), dpi=100)
    ax.plot(range(num_timesteps), noise_levels)
    if exp_name:
        ax.set_title(exp_name)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, num_timesteps - 1)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    fig.tight_layout()
    
    fig.subplots_adjust(left=0, right=1, bottom=0)
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_img = Image.open(buf)
    
    # Combine plot and image grid
    combined_img = Image.new('RGB', (image_grid_pil.width, image_grid_pil.height + plot_img.height))
    combined_img.paste(plot_img, (0, 0))
    combined_img.paste(image_grid_pil, (0, plot_img.height))
    
    return combined_img


def main(args):
    seed_everything(args.seed)
    
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu") if args.model_path else {}
    )
    model.to(device)
    model.eval()

    logger.log("sampling...")
    # model_kwargs = {}
    # if args.class_cond:
    #     classes = th.tensor([i for i in range(args.num_classes)] * 4, dtype=th.long, device=device)
    #     model_kwargs["y"] = classes
    # sample_fn = (
    #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    # )
    # sample = sample_fn(
    #     model,
    #     (args.batch_size, 3, args.image_size, args.image_size),
    #     clip_denoised=args.clip_denoised,
    #     model_kwargs=model_kwargs,
    #     progress=True
    # )
    # # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    # # sample = sample.permute(0, 2, 3, 1)
    # # sample = sample.contiguous()
    
    # display_image_grid(sample, display=True, nrow=10, normalize=True, value_range=(-1, 1), factor=args.factor)

    
    # classes = th.tensor([i for i in range(args.num_classes)], dtype=th.long, device=device)
    # model_kwargs["y"] = classes
    # sample_fn = (
    #     diffusion.p_sample_loop_progressive if not args.use_ddim else diffusion.ddim_sample_loop_progressive
    # )
    # x0_preds_progressive = []
    # for i, sample in enumerate(
    #     sample_fn(
    #         model,
    #         (args.num_classes, 3, args.image_size, args.image_size),
    #         clip_denoised=args.clip_denoised,
    #         model_kwargs=model_kwargs,
    #         progress=True
    #     )
    # ): 
    #     # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    #     # sample = sample.permute(0, 2, 3, 1)
    #     # sample = sample.contiguous()
    #     x0_preds_progressive.append(sample["pred_xstart"])
        
    # x0_preds_progressive = th.stack(x0_preds_progressive, dim=0)
    
    # # Select 10 evenly spaced timesteps, including first and last
    # indices = th.linspace(0, len(x0_preds_progressive) - 1, 10).long()
    # x0_preds_progressive = x0_preds_progressive[indices]
    
    # x0_preds_progressive = rearrange(x0_preds_progressive, 't b c h w -> (b t) c h w')
    
    # display_image_grid(x0_preds_progressive, display=True, nrow=10, normalize=True, value_range=(-1, 1), factor=args.factor)
    
    
    data = load_data(
        data_dir=args.data_dir,
        batch_size=5,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )
    
    batch, cond = next(iter(data))
    batch = batch.to(device)
    
    combined_im = timestamp_sample(
        diffusion,
        model,
        batch,
        model_kwargs=cond,
        use_ddim=args.use_ddim,
        num_timesteps=20,
        device=device,
        factor=args.factor,
        exp_name=args.exp_name,
    )
    
    from IPython.display import display as ipy_display
    ipy_display(combined_im)
        



#%%

# exp_name = "cifar10_uncond_openai"
exp_name = "cifar10_cond_openai"
# exp_name = "cifar10_cond_baseline"
# exp_name = "cifar10_cond_distributional_implementation_check"
# exp_name = "cifar10_cond_distributional_logsnr"
# exp_name = "cifar10_cond_distributional_noweighting"
# exp_name = "cifar10_cond_distributional_noweighting_lambda_linear"
# exp_name = "cifar10_cond_distributional_noweighting_eps_pred"
# exp_name = "cifar10_cond_distributional_noweighting_eps_pred"
checkpoint_iter = 300_000
ema = True
model = f"ema_0.9999_{checkpoint_iter}" if ema else f"model{checkpoint_iter}" 

defaults = dict(
    clip_denoised=False,
    num_classes=10,
    batch_size=10 * 4,
    image_size=32,
    num_channels=128,
    num_res_blocks=3,
    learn_sigma=True,
    diffusion_steps=4000,
    use_ddim=True,
    timestep_respacing="ddim500",
    class_cond="uncond" not in exp_name,
    predict_xstart=False,
    noise_schedule="cosine",
    lr=1e-4,
    model_path=f"/ceph/scratch/martorellat/guided_diffusion/blobs_{exp_name}/{model}.pt",
    factor=1,  # Factor to resize the image for display
    
    use_distributional="distributional" in exp_name,
    # distributional_lambda=0,
    # distributional_track_terms_regardless_of_lambda=True,
    # distributional_population_size=4,
    # distributional_kernel_kwargs='{"beta": 2}',
    data_dir="/nfs/ghome/live/martorellat/data/cifar_train",
    exp_name=exp_name,
    seed=10,
)
args = model_and_diffusion_defaults() | defaults

main(EasyDict(args))

# %%