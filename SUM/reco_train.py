import torch
from torch_radon import RadonFanbeam, Radon
import numpy as np
import argparse
from pathlib import Path
from .helper import load_tiff_stack_with_metadata, save_to_tiff_stack

def para_prepare_parallel(index):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_proj', type=str, default='/root/autodl-tmp/DDDM/fan_projections.tif', help='Local path of fan beam projection data.') #Reserve a tif file to quickly pass parameters
    parser.add_argument('--image_size', type=int, default=736, help='Size of reconstructed image.') # in line with detector number
    parser.add_argument('--voxel_size', type=float, default=0.7, help='In-slice voxel size [mm].')
    parser.add_argument('--fbp_filter', type=str, default='hann', nargs='?',choices=['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'], help='Filter used for FBP.')
    args = parser.parse_args()

    _, metadata = load_tiff_stack_with_metadata(Path(args.path_proj))

    vox_scaling = 1 / args.voxel_size 
    bias = 0 # To control the rotation angle of CT image
    angles = np.array(metadata['angles'])[:metadata['rotview']+bias] + (np.pi / 2)
    
    if index == 1:
        angles = angles[np.arange(0+bias, 1152+bias, index)]
    elif index == 16.5: #.5 means half-scan, integer part means the down-sample factor.
        angles = (angles[np.arange(0+bias, 1152+bias, 16)])[np.arange(0,36,1)]
    elif index == 8.5:
        angles = (angles[np.arange(0+bias, 1152+bias, 8)])[np.arange(0,72,1)]
    elif index == 4.5:
        angles = (angles[np.arange(0+bias, 1152+bias, 4)])[np.arange(0,144,1)]
    elif index == 2.5:
        angles = (angles[np.arange(0+bias, 1152+bias, 2)])[np.arange(0,288,1)]


    # radon = RadonFanbeam(args.image_size,
    #                          angles,
    #                          source_distance=vox_scaling * metadata['dso'],
    #                          det_distance=vox_scaling * metadata['ddo'],
    #                          det_count=736,
    #                          det_spacing=vox_scaling * metadata['du'],
    #                          clip_to_circle=False)
    radon = Radon(736, angles=angles, clip_to_circle=True)
    return radon

def run_reco(projections, radon): 
    # projections = projections[:,range_clip,:] # If FBP results are weired, try uncomment this line.
    if(len(projections.shape) == 4):
        sino = projections
    elif (len(projections.shape) == 3):
        sino = torch.flip(projections, dims=[2])
    elif (len(projections.shape) == 2):
        sino = torch.flip(projections, dims=[1])
    
    filtered_sinogram = radon.filter_sinogram(sino, filter_name='hann')
    fbp =  100 * radon.backprojection(filtered_sinogram)

    return fbp

