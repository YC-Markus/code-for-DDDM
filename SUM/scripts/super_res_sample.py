"""
Generate a single batch of samples from a super resolution model.
"""
import matplotlib.pyplot as plt
import argparse
import os
import torch.nn.functional as F
from improved_diffusion.image_datasets import improved_data_preprocess_val
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from skimage.metrics import structural_similarity as ssim
import lpips
from improved_diffusion.reco_train import run_reco, para_prepare_parallel, para_prepare

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


model_name = "ema_0.9999_028500"
parallel = True


def down_up(input,down_size,mode):
    output = F.interpolate(input, down_size, mode="nearest")
    output = F.interpolate(output, [288,736], mode=mode)
    return output


def p_sample_loop_super_res(model, batch_size, model_kwargs, raw_img):

    with th.no_grad():
        x_in =  F.interpolate(model_kwargs["low_res"], [288,736], mode="nearest")
        x2 = model(x_in, timesteps=th.tensor([2]).to("cuda"), **model_kwargs)
        x1 = model(x2, timesteps=th.tensor([1]).to("cuda"), **model_kwargs)
        x0 = model(x1, timesteps=th.tensor([0]).to("cuda"), **model_kwargs)
        x_out = x0
    return x_out


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    data = load_superres_data(args.batch_size, small_size=[36,736], class_cond=False)

    logger.log("creating samples...")
    all_images = []
    
    if parallel:
        radon_288_736 = para_prepare_parallel(2.5)
        radon_36_736 = para_prepare_parallel(16.5)

    helper = {"fbp_para_288_736": radon_288_736, "fbp_para_36_736": radon_36_736} 
    
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        raw_img = model_kwargs.pop('raw_img').to("cuda")
        raw_img = down_up(raw_img,[288,736],"nearest")


        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        model_kwargs["fbp_para_36_736"] = radon_36_736
        model_kwargs["fbp_para_288_736"] = radon_288_736
        
        input_fbp = run_reco(th.flip(model_kwargs['low_res'].to("cuda") + 1., dims=[3]), helper["fbp_para_36_736"])[:,:,112:624,112:624]
        input_npy = input_fbp.squeeze().cpu().detach().numpy()
        plt.imshow(input_npy, cmap=plt.cm.gray)

        
        sample_fn = p_sample_loop_super_res
        sample = sample_fn(
            model,
            (args.batch_size, 1, 288, 736), #args.large_size, args.large_size
            # clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            raw_img = raw_img,
        )
        
        
        model_output_fbp = run_reco(th.flip(sample + 1.,dims=[3]), helper["fbp_para_288_736"])[:,:,112:624,112:624]
        target_fbp = run_reco(th.flip(raw_img + 1., dims=[3]), helper["fbp_para_288_736"])[:,:,112:624,112:624]
        
        target_npy = target_fbp.squeeze().cpu().detach().numpy() 
        plt.imshow(target_npy, cmap=plt.cm.gray)

        npy = np.squeeze(model_output_fbp.cpu().detach().numpy())
        print("mean: ", np.mean(npy))
        print("std: ", np.std(npy))
        print("max: ", np.max(npy))
        print("min: ", np.min(npy))
        
        raw_npy = target_fbp.squeeze().cpu().detach().numpy()
        print("SSIM:", ssim(npy, raw_npy,data_range=raw_npy.max()-raw_npy.min()))
        
        l2loss = th.nn.MSELoss().to(dist_util.dev())
        print("MSELoss:", l2loss(model_output_fbp, target_fbp.to(dist_util.dev())).item())

        lpip_loss = lpips.LPIPS(net="alex").to(dist_util.dev())
        lpip_value = lpip_loss(model_output_fbp, target_fbp.to(dist_util.dev()))
        print("lpips:", lpip_value.item())
        
        break


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1,
        batch_size=1,
        use_ddim=True,
        base_samples="",
        model_path=f'/root/autodl-tmp/improved_diffusion/model_save_FULL/{model_name}.pt', # model065000 ema_0.9999_060000
    )
    print(defaults["model_path"])
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def load_superres_data(batch_size, small_size, class_cond=False):

    data = improved_data_preprocess_val(batch_size = batch_size, shuffle = True, num_workers = 8)
    radon_576_736 = para_prepare_parallel(2)
    
    for large_batch, model_kwargs in data:
        if parallel:
            _576_fbp = run_reco(th.flip(large_batch[:,:,np.arange(0, 1152, 2),:] + 1., dims=[3]).to("cuda"), radon_576_736)[:,:,112:624,112:624]
            
            
        plt.imshow(_576_fbp.squeeze().cpu().detach().numpy(), cmap=plt.cm.gray)        
        
        large_batch = large_batch[:,:,np.arange(0, 576, 2),:]
        npy = large_batch.squeeze().cpu().detach().numpy()
        print("mean: ", np.mean(npy))
        print("std: ", np.std(npy))
        print("max: ", np.max(npy))
        print("min: ", np.min(npy))
        
        model_kwargs["raw_img"] = large_batch
        model_kwargs["low_res"] = F.interpolate(large_batch, [36, 736], mode="nearest")

        res = dict(low_res=model_kwargs["low_res"], raw_img=model_kwargs["raw_img"])
        yield res
        
    
if __name__ == "__main__":
    main()
