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
import matplotlib.pyplot as plt
import lpips
from improved_diffusion.reco_train import run_reco, para_prepare_parallel


from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

def p_sample_loop_super_res(model, batch_size, model_kwargs):

    with th.no_grad():
        x_in =  F.interpolate(model_kwargs["low_res"], [288,736], mode="nearest")
        x2 = model(x_in, timesteps=th.tensor([2]).to("cuda"), **model_kwargs)
        x1 = model(x2, timesteps=th.tensor([1]).to("cuda"), **model_kwargs)
        x0 = model(x1, timesteps=th.tensor([0]).to("cuda"), **model_kwargs)
        x_out = x0
    return x_out, x2


def main():
    num = 800 # Total generated picture num
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cuda")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    
    data = load_superres_data(args.batch_size)

    logger.log("creating samples...")
    
    MSE, SSIM, PSNR, LPIP = [], [], [], []
    lpip_loss = lpips.LPIPS(net="alex").to(dist_util.dev())
    
    l2loss = th.nn.MSELoss().to(dist_util.dev())
    
    
    radon_288_736 = para_prepare_parallel(2.5)
    radon_72_736 = para_prepare_parallel(8.5)
    radon_36_736 = para_prepare_parallel(16.5)
    helper = {"fbp_para_288_736": radon_288_736, "fbp_para_36_736": radon_36_736, "fbp_para_72_736": radon_72_736} 
    
    for i in range(0, num//args.batch_size):#
        model_kwargs = next(data)
        raw_img = model_kwargs.pop('raw_img').to("cuda")
        index = model_kwargs.pop('index')
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        model_kwargs["fbp_para_36_736"] = radon_36_736
        model_kwargs["fbp_para_288_736"] = radon_288_736
        
        sample_fn = p_sample_loop_super_res
        sample, sample_72_288 = sample_fn(
            model,
            (args.batch_size, 1, 288, 736), #args.large_size, args.large_size
            # clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        
        model_72_sino = F.interpolate(sample_72_288, [72, 736], mode="nearest")
        model_72_fbp = run_reco(model_72_sino + 1., helper["fbp_para_72_736"])[:,:,112:624,112:624]
        model_72_fbp_npy = model_72_fbp.cpu().detach().numpy()
        
        model_output_fbp = run_reco(sample + 1., helper["fbp_para_288_736"])[:,:,112:624,112:624]
        target_fbp = run_reco(raw_img + 1., helper["fbp_para_288_736"])[:,:,112:624,112:624]
        output_fbp_npy = model_output_fbp.cpu().detach().numpy()
 
        for j in range(0, args.batch_size):

            l2loss_value = l2loss(model_output_fbp[j], target_fbp[j]).item()
            print("index:", index[j], "MSELoss:", l2loss_value)
            MSE.append(l2loss_value)

            raw_npy = target_fbp.cpu().detach().numpy()
            ssim_value = ssim(np.squeeze(output_fbp_npy[j]),np.squeeze( raw_npy[j]), data_range = raw_npy[j].max() - raw_npy[j].min())
            psnr_value = psnr(np.squeeze(output_fbp_npy[j]),np.squeeze( raw_npy[j]), data_range = raw_npy[j].max() - raw_npy[j].min())
            print("index:", index[j], "SSIM:", ssim_value)
            SSIM.append(ssim_value)
            PSNR.append(psnr_value)

            lpip_value = lpip_loss(model_output_fbp[j], target_fbp[j])
            print("lpips:", lpip_value.item())
            LPIP.append(lpip_value.item())
        
    print("mse mean:", np.mean(MSE))
    print("ssim mean:", np.mean(SSIM))
    print("psnr mean:", np.mean(PSNR))
    print("lpip mean:", np.mean(LPIP))
    

def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1,
        batch_size=2,
        use_ddim=True,
        base_samples="",
        model_path='/root/autodl-tmp/TASK8/improved_diffusion/model_save_FULL/ema_0.9999_036000.pt',
    )
    print(defaults["model_path"])
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_superres_data(batch_size):
    
    data = trans_data_preprocess(batch_size = batch_size, shuffle = False, num_workers = 8)
    
    for large_batch, model_kwargs in data:
        
        large_batch = large_batch[:,:,np.arange(0, 576, 2),:]

        model_kwargs["raw_img"] = large_batch
        model_kwargs["low_res"] = F.interpolate(large_batch, [36, 736], mode="nearest") #36->72
        
        res = dict(low_res=model_kwargs["low_res"], raw_img=model_kwargs["raw_img"],index=model_kwargs["index"] )
        yield res
        
    
if __name__ == "__main__":
    main()