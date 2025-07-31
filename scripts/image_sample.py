"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
from io import BytesIO
import os

from einops import rearrange, repeat
from matplotlib import pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

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

def sample_progressive_images(
    args, model, model_kwargs, diffusion, device, save_path=None
):
    if args.class_cond:
        classes = th.tensor([i for i in range(args.num_classes)], dtype=th.long, device=device)
        model_kwargs["y"] = classes
    sample_fn = (
        diffusion.p_sample_loop_progressive if not args.use_ddim else diffusion.ddim_sample_loop_progressive
    )
    x0_preds_progressive = []
    for i, sample in enumerate(
        sample_fn(
            model,
            (args.num_classes, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
    ): 
        x0_preds_progressive.append(sample["pred_xstart"])
        
    x0_preds_progressive = th.stack(x0_preds_progressive, dim=0)
    
    # Select 10 evenly spaced timesteps, including first and last
    indices = th.linspace(0, len(x0_preds_progressive) - 1, 10).long()
    x0_preds_progressive = x0_preds_progressive[indices]
    
    x0_preds_progressive = rearrange(x0_preds_progressive, 't b c h w -> (b t) c h w')

    save_path = f"{save_path}/x0_preds_progressive.png"
    display_image_grid(x0_preds_progressive, save_path=save_path, nrow=10, normalize=True, value_range=(-1, 1), factor=args.vis_magnification)


def timestamp_sample(
    diffusion,
    model,
    imgs,
    model_kwargs=None,
    use_ddim=False,
    num_timesteps=50,
    device=None,
    factor=1,
    exp_name=None,
    save_path=None
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
        eps = eps[:, :diffusion.distributional_num_eps_channels, ...]
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
    
    if save_path:
        combined_img.save(f"{save_path}/xstart_preds_given_gt_noisy.png")

    return combined_img

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    
    num_batches = int(np.ceil(args.num_samples / (args.batch_size * dist.get_world_size())))
    
    pbar = range(num_batches)
    if dist.get_rank() == 0:
        pbar = tqdm(pbar, desc=f"Sampling {args.num_samples} images")

    for _ in pbar:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        if os.environ.get("OPENAI_SAMPLESDIR") is not None:
            base_path = os.path.join(os.environ["OPENAI_SAMPLESDIR"])
        else:
            base_path = os.path.join(logger.get_dir())
        os.makedirs(base_path, exist_ok=True)
        out_path = os.path.join(base_path, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
            
            # now we save 50 images for visualization but we pick 5 for each class
            selected_imgs = []
            num_images_per_class = 5
            for i in range(args.num_classes):
                class_indices = np.where(label_arr == i)[0]
                selected_indices = class_indices[:num_images_per_class]
                selected_imgs.append(arr[selected_indices])
            
            selected_imgs = np.concatenate(selected_imgs, axis=0)
            imgs = th.from_numpy(selected_imgs).permute(0, 3, 1, 2).float() / 255.0
            display_image_grid(imgs, save_path=os.path.join(base_path, f"grid_{shape_str}.png"), factor=4, nrow=10)
            
        else:
            np.savez(out_path, arr)
            
            # now we just save the first 50 images for visualization
            imgs = th.from_numpy(arr[:50]).permute(0, 3, 1, 2).float() / 255.0
            display_image_grid(imgs, save_path=os.path.join(base_path, f"grid_{shape_str}.png"), nrow=10, factor=4)

        # here we do a progressive sampling of the images and show the x0 predictions
        sample_progressive_images(
            args, model, model_kwargs, diffusion, dist_util.dev(), save_path=base_path
        )
        
        # Here we do a timestamped sampling of the images and show the x0 predictions
        data = load_data(
            data_dir=args.data_dir,
            batch_size=5,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
        )
        
        batch, cond = next(iter(data))
        batch = batch.to(dist_util.dev())

        timestamp_sample(
            diffusion,
            model,
            batch,
            model_kwargs=cond,
            use_ddim=args.use_ddim,
            num_timesteps=20,
            device=dist_util.dev(),
            factor=args.vis_magnification,
            exp_name=args.exp_name,
            save_path=base_path
        )

    dist.barrier()
    dist_util.cleanup()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_classes=10,
        image_size=32,
        num_channels=128,
        num_res_blocks=3,
        predict_xstart=True,
        class_cond=True,
        diffusion_steps=4000,
        use_distributional=False,
        num_samples=10000,
        batch_size=256,
        use_ddim=True,
        timestep_respacing="ddim50",
        model_path="",
        vis_magnification=4,
        exp_name="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
