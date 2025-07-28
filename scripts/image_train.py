"""
Train a diffusion model on images.
"""

import argparse
from datetime import datetime
import torch.distributed as dist
import os

from guided_diffusion import dist_util, logger, wandb_utils
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    seed_everything,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()
    # seed_everything(args.seed)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model_and_diffusion_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_args
    )
    _, sampling_diffusion = create_model_and_diffusion(
        **(model_and_diffusion_args | {
            "timestep_respacing": "250",
        })
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
    if dist.get_rank() == 0:     
        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        experiment_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}-{os.environ['EXPERIMENT_NAME']}"
        wandb_utils.initialize(args, entity, experiment_name, project)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        sampling_diffusion=sampling_diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_steps=args.max_steps,  # New argument for maximum steps
    ).run_loop()
    
    wandb_utils.cleanup()
    dist_util.cleanup()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        sample_interval=10000,
        max_steps=300000,  # Default to 300k steps
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=42,  # Default seed for reproducibility
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
