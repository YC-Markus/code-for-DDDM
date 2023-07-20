from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import os
import random
import torch.nn.functional as F


    
def improved_data_preprocess(sino_path = '/root/autodl-tmp/TASK4/SINO_DATA/val_stage1', batch_size = 1, shuffle = True, num_workers = 8):
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
        return 3240 + 4140 +3260 + 3260 #self.sino_num # C + L 3254 + 4152
    def __getitem__(self,index):
        out_dict = {}
        if index<3240:
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load(self.sino_path + f'/full_C_parallel_img/{index}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            # x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_C_parallel_sino/{index}.npy')).to(torch.float32),dim=0)
            out_dict["low_res"] = torch.tensor(np.load(self.sino_path + f'/36to288_C_parallel_img/{index}.npy')).to(torch.float32)
            out_dict["_36_res"] = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_C_parallel_sino/{index}.npy')).to(torch.float32),dim=0)
            # out_dict["y"] = np.array(0, dtype=np.int64)
            
        elif index<(3240 + 4140):
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load(self.sino_path + f'/full_L_parallel_img/{index - 3240}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            # x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index - 3254}.npy')).to(torch.float32),dim=0)
            out_dict["low_res"] = torch.tensor(np.load(self.sino_path + f'/36to288_L_parallel_img/{index - 3240}.npy')).to(torch.float32)
            out_dict["_36_res"] = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index - 3240}.npy')).to(torch.float32),dim=0)
            # out_dict["y"] = np.array(1, dtype=np.int64)

        elif index < (3240 + 4140 + 3260):
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/full_C_parallel_img/{index - 3240 - 4140}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            out_dict["low_res"] = torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/36to288_C_parallel_img/{index - 3240 - 4140}.npy')).to(torch.float32)
            out_dict["_36_res"] = torch.unsqueeze(torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/full_C_parallel_sino/{index - 3240 - 4140}.npy')).to(torch.float32),dim=0)

        else:
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/full_L_parallel_img/{index - 3240 - 4140 - 3260}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            out_dict["low_res"] = torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/36to288_L_parallel_img/{index - 3240 - 4140 - 3260}.npy')).to(torch.float32)
            out_dict["_36_res"] = torch.unsqueeze(torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/full_L_parallel_sino/{index - 3240 - 4140 - 3260}.npy')).to(torch.float32),dim=0)

        # x_nd_mean = torch.mean(x_nd)
        # x_nd = x_nd - x_nd_mean
        # x_nd_min_abs = torch.abs(torch.min(x_nd))
        # x_nd = x_nd / x_nd_min_abs
    
        # out_dict["low_res"] = out_dict["low_res"] - x_nd_mean # torch.mean(out_dict["low_res"]) 
        # out_dict["low_res"] = out_dict["low_res"] / x_nd_min_abs # torch.abs(torch.min(out_dict["low_res"]))
        # print("x_nd:", torch.min(x_nd))
        # print("out_dict:", torch.min(out_dict["low_res"]))
        # print("out_dict:", torch.mean(out_dict["low_res"]))

        assert len(x_nd.shape) == 3
        return x_nd, out_dict
    

    
    
    
def improved_data_preprocess_val(sino_path = '/root/autodl-tmp/TASK4/SINO_DATA/val_stage2', batch_size = 1, shuffle = False, num_workers = 8):
    train_dataset = diffCT_Dataset_val(sino_path)
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    while True:
        yield from train_data_loader
    
    
class diffCT_Dataset_val(Dataset):
    def __init__(self, sino_path):
        self.sino_path = sino_path
        self.sino_num = count_npy_files(sino_path)
    def __len__(self):
        return 3240 + 4140 +3260 + 3260 #self.sino_num # C + L
    def __getitem__(self,index):
        out_dict = {}
        index = 150
        if index<3240:
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load(self.sino_path + f'/full_C_parallel_img/{index}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            # x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_C_parallel_sino/{index}.npy')).to(torch.float32),dim=0)
            out_dict["low_res"] = torch.tensor(np.load(self.sino_path + f'/36to288_C_parallel_img/{index}.npy')).to(torch.float32)
            # out_dict["_72_res"] = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_C_parallel_sino/{index}.npy')).to(torch.float32),dim=0)
            # out_dict["y"] = np.array(0, dtype=np.int64)
            
        elif index<(3240 + 4140):
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load(self.sino_path + f'/full_L_parallel_img/{index - 3240}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            # x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index - 3254}.npy')).to(torch.float32),dim=0)
            out_dict["low_res"] = torch.tensor(np.load(self.sino_path + f'/36to288_L_parallel_img/{index - 3240}.npy')).to(torch.float32)
            # out_dict["_72_res"] = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index - 3254}.npy')).to(torch.float32),dim=0)
            # out_dict["y"] = np.array(1, dtype=np.int64)

        elif index < (3240 + 4140 + 3260):
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/full_C_parallel_img/{index - 3240 - 4140}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            out_dict["low_res"] = torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/36to288_C_parallel_img/{index - 3240 - 4140}.npy')).to(torch.float32)

        else:
            x_nd = torch.flip(torch.unsqueeze(100. * torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/full_L_parallel_img/{index - 3240 - 4140 - 3260}.npy')[112:624,112:624]).to(torch.float32),dim=0),dims=[1,2])
            out_dict["low_res"] = torch.tensor(np.load('/root/autodl-tmp/TASK4/SINO_DATA/train_stage1' + f'/36to288_L_parallel_img/{index - 3240 - 4140 - 3260}.npy')).to(torch.float32)
            
        assert len(x_nd.shape) == 3
        return x_nd, out_dict
    
    
    
    
def trans_data_preprocess(sino_path = '/root/autodl-tmp/TASK4/SINO_DATA/val_stage2', batch_size = 1, shuffle = False, num_workers = 8):
    train_dataset = trans_diffCT_Dataset(sino_path)
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    while True:
        yield from train_data_loader
        
        
class trans_diffCT_Dataset(Dataset):
    def __init__(self, sino_path):
        self.sino_path = sino_path
        self.sino_num = count_npy_files(sino_path)
    def __len__(self):
        # return 2020-496
        return  800# -496
    def __getitem__(self,index):
        out_dict = {}
        index = index +0# +496 +450/98 +103
        x_nd = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index}.npy')).to(torch.float32),dim=0)
        out_dict["low_res"] = torch.tensor(np.load(self.sino_path + f'/test_pic_72/{index}.npy')).to(torch.float32) # 36to288_C_parallel_img test_pic # 36to288_L_parallel_img
        out_dict["_36_res"] = torch.unsqueeze(torch.tensor(np.load(self.sino_path + f'/full_L_parallel_sino/{index}.npy')).to(torch.float32),dim=0)
            
        # out_dict["y"] = np.array(1, dtype=np.int64)

        out_dict["index"] = index
        assert len(x_nd.shape) == 3
        return x_nd, out_dict