import torch
from torch_radon import RadonFanbeam, Radon
import numpy as np
import argparse
from pathlib import Path
from .helper import load_tiff_stack_with_metadata, save_to_tiff_stack


def para_prepare_1(index):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_proj', type=str, default='/root/DDDM/fan_projections.tif', help='Local path of fan beam projection data.')
    parser.add_argument('--image_size', type=int, default=736, help='Size of reconstructed image.')
    parser.add_argument('--voxel_size', type=float, default=0.7, help='In-slice voxel size [mm].')
    parser.add_argument('--fbp_filter', type=str, default='hann', nargs='?',choices=['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'], help='Filter used for FBP.')
    args = parser.parse_args()

    _, metadata = load_tiff_stack_with_metadata(Path(args.path_proj))

    vox_scaling = 1 / args.voxel_size 
    angles = np.array(metadata['angles'])[:metadata['rotview']] + (np.pi / 2)
    if index == 4:
        angles = angles[np.arange(3, 1155, index)]
    elif index == 1:
        angles = angles[np.arange(0, 1152, index)]
    elif index == 16:
        angles = angles[np.arange(15, 1167, index)]
    elif index == 2.5:
        angles = (angles[np.arange(1, 1153, 2)])[np.arange(1,289,1)]
    elif index == 2:
        angles = angles[np.arange(1, 1153, 2)]
    radon = RadonFanbeam(args.image_size,
                             angles,
                             source_distance=vox_scaling * metadata['dso'],
                             det_distance=vox_scaling * metadata['ddo'],
                             det_count=736,
                             det_spacing=vox_scaling * metadata['du'],
                             clip_to_circle=True)
    return radon


def para_prepare_parallel(index):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_proj', type=str, default='/root/autodl-tmp/TASK2/fan_projections.tif', help='Local path of fan beam projection data.')
    parser.add_argument('--image_size', type=int, default=736, help='Size of reconstructed image.')
    parser.add_argument('--voxel_size', type=float, default=0.7, help='In-slice voxel size [mm].')
    parser.add_argument('--fbp_filter', type=str, default='hann', nargs='?',choices=['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'], help='Filter used for FBP.')
    args = parser.parse_args()

    _, metadata = load_tiff_stack_with_metadata(Path(args.path_proj))

    vox_scaling = 1 / args.voxel_size 
    bias = 0# 103 180
    angles = np.array(metadata['angles'])[:metadata['rotview']+bias] + (np.pi / 2)
    if index == 1.01:
        angles = angles[np.arange(0+bias, 1152+bias, 1)]
    elif index == 16.5:
        angles = (angles[np.arange(0+bias, 1152+bias, 16)])[np.arange(0,36,1)]
    elif index == 8.5:
        angles = (angles[np.arange(0+bias, 1152+bias, 16)])[np.arange(0,18,1)]

    # radon = RadonFanbeam(args.image_size,
    #                          angles,
    #                          source_distance=vox_scaling * metadata['dso'],
    #                          det_distance=vox_scaling * metadata['ddo'],
    #                          det_count=736,
    #                          det_spacing=vox_scaling * metadata['du'],
    #                          clip_to_circle=False)
    radon = Radon(736, angles=angles, clip_to_circle=True) # det_spacing=vox_scaling * metadata['du'], det_count=736
    return radon

def run_reco(projections, radon): 
    # projections = projections[:,range_clip,:]
    if(len(projections.shape) == 4):
        sino = torch.flip(projections, dims=[3]) 
    elif (len(projections.shape) == 3):
        sino = torch.flip(projections, dims=[2])
    elif (len(projections.shape) == 2):
        sino = torch.flip(projections, dims=[1])
    
    filtered_sinogram = radon.filter_sinogram(sino, filter_name='hann')
    fbp =  100 * radon.backprojection(filtered_sinogram)

    return fbp

