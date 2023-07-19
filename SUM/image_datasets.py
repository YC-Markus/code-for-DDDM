from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import os
import random
import torch.nn.functional as F


    
def improved_data_preprocess(sino_path = '/root/autodl-tmp/TASK4/SINO_DATA/train_stage1', batch_size = 1, shuffle = True, num_workers = 8):
    train_dataset = diffCT_Dataset(sino_path)
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    while True:
        yield from train_data_loader
    
def count_npy_files(folder_path: str):
    files = os.listdir(folder_path)
    npy_files = [file for file in files if file.endswith('.npy')]
    return len(npy_files)
    
class diffCT_Dataset(Dataset):
    def __init__(self, sino_path):
        self.sino_path = sino_path
        self.sino_num = count_npy_files(sino_path)
        
    def __len__(self):
        return (3276+3276+3255+4154) #self.sino_num
    def __getitem__(self,index):
        out_dict = {}

        if index < (3276+3276) :
            x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_C_parallel_sino/{index}.npy')).to(torch.float32),dim=0) - 1.
            # x_nd = torch.tensor(np.load(self.sino_path + f'/full_C_fan_sino/{index}.npy')).to(torch.float32)
            # out_dict["y"] = np.array(0, dtype=np.int64)
            
        elif index < (3255+4154):
            x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index-3276}.npy')).to(torch.float32),dim=0) - 1.
            # x_nd = torch.tensor(np.load(self.sino_path + f'/full_L_fan_sino/{index-3276}.npy')).to(torch.float32)
            # out_dict["y"] = np.array(1, dtype=np.int64)

        return x_nd, out_dict
    

    
    
    
def improved_data_preprocess_val(sino_path = '/root/autodl-tmp/TASK4/SINO_DATA/train_stage1', batch_size = 1, shuffle = True, num_workers = 8):
    train_dataset = diffCT_Dataset_val(sino_path)
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    while True:
        yield from train_data_loader
    
class diffCT_Dataset_val(Dataset):
    def __init__(self, sino_path):
        self.sino_path = sino_path
        self.sino_num = count_npy_files(sino_path)
    def __len__(self):
        return 1000 #self.sino_num
    def __getitem__(self,index):
        out_dict = {}

        choice1 = random.random()
        index = 100
        if choice1 < (1/3):
            x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_C_parallel_sino/{index}.npy')).to(torch.float32),dim=0) - 1.
            # out_dict["y"] = np.array(0, dtype=np.int64)
            
        elif choice1 < (2/3):
            x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index}.npy')).to(torch.float32),dim=0) - 1.
            # out_dict["y"] = np.array(1, dtype=np.int64)
            
        else:
            x_nd = torch.tensor(np.load(f'/root/autodl-tmp/TASK4/SINO_DATA/val_stage2/full_L/val_{index}.npy')).to(torch.float32)
            # out_dict["y"] = np.array(2, dtype=np.int64)

        return x_nd, out_dict
    
    
    
    
def trans_data_preprocess(sino_path = '/root/autodl-tmp/TASK4/SINO_DATA/val_stage2', batch_size = 1, shuffle = False, num_workers = 8):
    train_dataset = trans_diffCT_Dataset(sino_path)
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    while True:
        yield from train_data_loader

    
class trans_diffCT_Dataset(Dataset):
    def __init__(self, sino_path):
        self.sino_path = sino_path
        # self.sino_num = count_npy_files(sino_path)
    def __len__(self):
        return 800 #-496
    def __getitem__(self,index):
        out_dict = {}
        index = index + 0
        x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index}.npy')).to(torch.float32),dim=0) - 1.

        # out_dict["y"] = np.array(1, dtype=np.int64)
        out_dict["index"] = index
            
        return x_nd, out_dict
