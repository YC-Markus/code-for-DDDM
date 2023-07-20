"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F
import torch
import numpy as np

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import improved_data_preprocess
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.reco_train import run_reco, para_prepare_parallel


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    radon_36 = para_prepare_parallel(16.5)
    radon_1152 = para_prepare_parallel(1.01)
    
    data = load_superres_data(
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        radon_36=radon_36,
        radon_1152=radon_1152
    )
    
    batch, cond = next(data)
    print(batch.shape)
    print(cond["low_res"].shape)
    
    for i in range(0,10):
        batch, cond = next(data)
        print(torch.max(batch[0]))
        print(torch.min(batch[0]))
        print(torch.mean(batch[0]))
        print(torch.max(cond["low_res"][0]))
        print(torch.min(cond["low_res"][0]))
        print(torch.mean(cond["low_res"][0]))
        print("                   ")
    
    
    logger.log("training...")
    print(args)
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def norm(fbp_img):
    return (fbp_img - (-0.75))

def load_superres_data(batch_size, large_size, small_size, class_cond, radon_36, radon_1152):
    print("class_cond:", class_cond)
    data = improved_data_preprocess(batch_size = batch_size, shuffle = True, num_workers = 8)
    for large_batch, model_kwargs in data:
        large_batch = norm(large_batch.to("cuda") - 1.)
        model_kwargs["low_res"] = norm(model_kwargs["low_res"].to("cuda") - 1.)
        model_kwargs["_36_res"] = norm(torch.flip(run_reco(model_kwargs["_36_res"][:,:,np.arange(0,576,16),:].to("cuda"), radon_36) - 1.,dims=[2,3])[:,:,112:624,112:624])
        
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="loss-second-moment",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=250,
        save_interval=2500,
        resume_checkpoint="/root/autodl-tmp/TASK7/improved_diffusion/model_save_3channels_modi/model010000.pt",# 
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
