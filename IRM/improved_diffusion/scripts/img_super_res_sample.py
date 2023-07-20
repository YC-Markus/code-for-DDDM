"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
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
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from pytorch_msssim import ms_ssim, MS_SSIM, SSIM


from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.reco_train import run_reco, para_prepare_parallel


index = 1
model_name = "model100000"

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
    
    radon_72 = para_prepare_parallel(16.5)
    radon_1152 = para_prepare_parallel(1.01)
    
    
    logger.log("loading data...")
    data = load_superres_data(
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        radon_72=radon_72,
        radon_1152=radon_1152
    )

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        raw_img, model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        
        npy_input = model_kwargs['low_res'].squeeze().cpu().detach().numpy()
        np.save(f"/root/autodl-tmp/TASK7/improved_diffusion/test_pic/{index}_{model_name}_input.npy", npy_input)
        plt.imshow(npy_input, cmap=plt.cm.bone)
        plt.savefig(f"/root/autodl-tmp/TASK7/improved_diffusion/test_pic/{index}_{model_name}_input.png")
        
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        
        sample = sample_fn(
            model,
            (args.batch_size, 1, 512, 512), #args.large_size, args.large_size
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        npy = sample.squeeze().cpu().detach().numpy()

        plt.imshow(npy, cmap=plt.cm.bone)
        
        l2loss = th.nn.MSELoss().to(dist_util.dev())
        print("MSELoss:", l2loss(sample, raw_img.to(dist_util.dev())).item())
        raw_npy = raw_img.squeeze().cpu().detach().numpy()
        plt.imshow(raw_npy, cmap=plt.cm.bone)
        np.save(f"/root/autodl-tmp/TASK7/improved_diffusion/test_pic/{index}_{model_name}_raw.npy", raw_npy)
        plt.savefig(f"/root/autodl-tmp/TASK7/improved_diffusion/test_pic/{index}_{model_name}_raw.png")
        print("PSNR:", psnr(npy, raw_npy, data_range=np.max(raw_npy)-np.min(raw_npy)))
        
        _ms_ssim = SSIM(win_size=7, win_sigma=1.5, data_range=1, size_average=False, channel=1)
        print("MS_SSIM:", _ms_ssim((sample+1)/4, ((raw_img+1)/4).to(dist_util.dev())))
        print("SSIM:", ssim(npy, raw_npy, data_range=raw_npy.max()-raw_npy.min())) # 
        lpip_loss = lpips.LPIPS(net="alex").to(dist_util.dev())

        print("max sample:", th.max(sample).item())
        print("min sample:", th.min(sample).item())
        print("max raw_img:", th.max(raw_img).item())
        print("min raw_img:", th.min(raw_img).item())
        lpip_value = lpip_loss(sample, raw_img.to(dist_util.dev()))
        print("lpips:", lpip_value.item())
        
        
        print("MSELoss input:", l2loss(model_kwargs['low_res'], raw_img.to(dist_util.dev())).item())
        print("SSIM input:", ssim(npy_input, raw_npy, data_range=raw_npy.max()-raw_npy.min()))
        print("PSNR input:", psnr(npy_input, raw_npy, data_range=raw_npy.max()-raw_npy.min()))
        lpip_value = lpip_loss(model_kwargs['low_res'], raw_img.to(dist_util.dev()))
        print("lpips input:", lpip_value.item())
        
        break
        



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=True,
        base_samples="",
        model_path=f'/root/autodl-tmp/TASK7/improved_diffusion/model_save_noise/{model_name}.pt',
    )
    print(defaults["model_path"])
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def norm(fbp_img):
    return ((fbp_img - (-0.75)) / 1.5) #-0.75 is the average value of whole CT images, 1.5 is the factor that make black area linearly mapping to -1


def load_superres_data(batch_size, large_size, small_size, class_cond, radon_72, radon_1152):
    print("class_cond:", class_cond)
    data = improved_data_preprocess_val(batch_size = batch_size, shuffle = True, num_workers = 8)
    for large_batch, model_kwargs in data:
        # large_batch = norm(large_batch.to("cuda") - 1.)
        large_batch = norm(large_batch.to("cuda") - 1.)
        model_kwargs["low_res"] = norm(model_kwargs["low_res"].to("cuda") - 1.)
        # model_kwargs["_72_res"] = norm(torch.flip(run_reco(F.interpolate((model_kwargs["_72_res"][:,:,np.arange(0,576,2),:] ).to("cuda"), (72, 736), mode="nearest"), radon_72) - 1.,dims=[2,3])[:,:,112:624,112:624])
        
        yield large_batch, model_kwargs
    
if __name__ == "__main__":
    main()
