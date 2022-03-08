import numpy as np
import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage import zoom, binary_dilation
import torch.nn.functional as F

def read_txt(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    name_list = [l.strip() for l in lines]
    return name_list

class ZoomedNpyDataset(Dataset):
    def __init__(self, img_root, msk_root, txtfile, istrain):
        self.istrain = istrain
        self.name_list = read_txt(txtfile)
        self.img_root = img_root
        self.msk_root = msk_root

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        img = np.load(os.path.join(self.img_root, name))
        msk = np.load(os.path.join(self.msk_root, name))
        # img = zoom(img, zoom=(0.5, 0.5, 0.5), order=0)
        # msk = binary_dilation(msk, structure=np.ones((3,3,3)), iterations=1)
        # msk = zoom(msk, zoom=(0.5, 0.5, 0.5), order=0)
        msk = (msk > 0).astype(int)
        img[img < -500] = -500
        img[img > 1000] = 1000
        img = (img - img.min()) / (img.max() - img.min())
        img = img[np.newaxis, np.newaxis, ...]
        msk = msk[np.newaxis, np.newaxis, ...].astype(float)
        img = F.interpolate(torch.from_numpy(img), scale_factor=0.7, mode='trilinear', align_corners=True, recompute_scale_factor=True)
        msk = F.interpolate(torch.from_numpy(msk), scale_factor=0.7, mode='nearest', recompute_scale_factor=True)
        msk[msk >= 0.1] = 1
        msk[msk < 0.1] = 0

        return img.squeeze(0), msk.squeeze(0)

class NiiTestDataset(Dataset):
    def __init__(self, img_root, msk_root, txtfile, istrain):
        self.istrain = istrain
        self.name_list = read_txt(txtfile)
        self.img_root = img_root
        self.msk_root = msk_root

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        img = sitk.ReadImage(os.path.join(self.img_root, name))
        img = sitk.GetArrayFromImage(img)
        shape_ori = img.shape
        img = zoom(img, zoom=(256 / img.shape[0], 256 / img.shape[1], 256 / img.shape[2]), order=0)
        # msk = binary_dilation(msk, structure=np.ones((3,3,3)), iterations=1)
        # msk = zoom(msk, zoom=(0.5, 0.5, 0.5), order=0)
        img[img < -500] = -500
        img[img > 1000] = 1000
        img = (img - img.min()) / (img.max() - img.min())
        img = img[np.newaxis, np.newaxis, ...]
        img = F.interpolate(torch.from_numpy(img), scale_factor=0.7, mode='trilinear', align_corners=True, recompute_scale_factor=True)

        return img.squeeze(0), np.array(shape_ori)







