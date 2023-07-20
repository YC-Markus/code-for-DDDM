"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""
import matplotlib.pyplot as plt
import argparse
import os
import torch.nn.functional as F
from improved_diffusion.image_datasets import trans_data_preprocess
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from improved_diffusion.reco_train import run_reco, para_prepare_parallel
import PIL.Image as Image


from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

model_name = "ema_0.9999_062500"

def main():
    num = 800 # total image numbers
    
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
    
    radon_36 = para_prepare_parallel(16.5)
    radon_72 = para_prepare_parallel(8.5)
    radon_1152 = para_prepare_parallel(1.01)
    data = load_superres_data(args.batch_size, radon_36=radon_36, radon_1152=radon_1152)
    
    
    logger.log("creating samples...")
    
    MSE, SSIM, PSNR, LPIP = [], [], [], []
    RAW, RECON = [], []
    MSE_36, SSIM_36, PSNR_36, LPIP_36 = [], [], [], []
    lpip_loss = lpips.LPIPS(net="alex").to(dist_util.dev())
    l2loss = th.nn.MSELoss().to(dist_util.dev())
    
    
    
    for i in range(0, num//args.batch_size):#
        raw_img, model_kwargs = next(data)
        index = model_kwargs.pop('index')
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, 512, 512),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        
        sample = sample + 10 # make sure samples are all positive
        raw_img = raw_img + 10


        model_kwargs["_36_res"] = model_kwargs["_36_res"] + 10

        assert sample.min()>0 and raw_img.min()>0
        output_fbp_npy = sample.cpu().detach().numpy().squeeze()
        RAW.append(th.mean(raw_img).item())
        RECON.append(th.mean(sample).item())
 
        for j in range(0, args.batch_size):      
            
            raw_npy = raw_img.cpu().detach().numpy()

            l2loss_value = l2loss(sample[j], raw_img[j]).item()
            print("index:", index[j].item(), "MSELoss:", l2loss_value)
            MSE.append(l2loss_value)
            
            ssim_value = ssim(np.squeeze(output_fbp_npy[j]),np.squeeze(raw_npy[j]), data_range = raw_npy[j].max() - raw_npy[j].min())
            print("index:", index[j].item(), "SSIM:", ssim_value)
            SSIM.append(ssim_value)
            psnr_value = psnr(np.squeeze(output_fbp_npy[j]),np.squeeze(raw_npy[j]), data_range = raw_npy[j].max() - raw_npy[j].min())
            PSNR.append(psnr_value)

            lpip_value = lpip_loss(sample[j] - 10, raw_img[j] - 10)
            print("lpips:", lpip_value.item())
            LPIP.append(lpip_value.item())

            ssim_value_36 = ssim((model_kwargs["_36_res"][j]).cpu().detach().numpy().squeeze(),np.squeeze(raw_npy[j]),data_range = raw_npy[j].max() - raw_npy[j].min()) # [:,100:412,100:412]
            SSIM_36.append(ssim_value_36)
            l2loss_value_36 = l2loss((model_kwargs["_36_res"][j]), raw_img[j]).item()
            MSE_36.append(l2loss_value_36)
            psnr_value_36 = psnr((model_kwargs["_36_res"][j]).cpu().detach().numpy().squeeze(),np.squeeze(raw_npy[j]),data_range = raw_npy[j].max() - raw_npy[j].min())
            PSNR_36.append(psnr_value_36)
            lpip_value_36 = lpip_loss((model_kwargs["_36_res"][j]) - 10, raw_img[j] - 10)
            LPIP_36.append(lpip_value_36.item())

        
    print("mse mean:", np.mean(MSE))
    print("ssim mean:", np.mean(SSIM))
    print("lpip mean:", np.mean(LPIP))
    print("psnr mean:", np.mean(PSNR))
    print("raw mean:", np.mean(RAW))
    print("recon mean:", np.mean(RECON))

    print("mse mean:", np.mean(MSE_36)) # metrics for original 36-view CT image
    print("ssim mean:", np.mean(SSIM_36))
    print("lpip mean:", np.mean(LPIP_36))
    print("psnr mean:", np.mean(PSNR_36))
    

def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1,
        batch_size=10,
        use_ddim=True,
        base_samples="",
        model_path=f'/root/autodl-tmp/IRM/improved_diffusion/model_save_3channels/{model_name}.pt',
    )
    print(defaults["model_path"])
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def norm(fbp_img):
    return (fbp_img - (-0.75))

def load_superres_data(batch_size, radon_36, radon_1152):
    
    data = trans_data_preprocess(batch_size = batch_size, shuffle = False, num_workers = 8)
    
    for large_batch, model_kwargs in data:
        # large_batch = norm(large_batch.to("cuda") - 1.)
        large_batch = norm(th.flip(run_reco((large_batch).to("cuda"), radon_1152) - 1.,dims=[2,3])[:,:,112:624,112:624])
        model_kwargs["low_res"] = norm(model_kwargs["low_res"].to("cuda") - 1.)
        model_kwargs["_36_res"] = norm(th.flip(run_reco(model_kwargs["_36_res"][:,:,np.arange(0,576,8),:].to("cuda"), radon_36) - 1.,dims=[2,3])[:,:,112:624,112:624])
        _36_npy = model_kwargs["_36_res"].cpu().detach().numpy().squeeze()
        yield large_batch, model_kwargs
        
    
if __name__ == "__main__":
    main()
