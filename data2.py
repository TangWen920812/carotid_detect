import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
import torch.nn.functional as F

def random_flip(data_list):
    flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    for i in range(len(data_list)):
        data_list[i] = np.ascontiguousarray(data_list[i][::flipid[0], ::flipid[1], ::flipid[2]])
    return data_list

def random_rotate(data_list):
    def rotate_left(data):
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data
    def rotate_right(data):
        data = np.ascontiguousarray(data[:, ::-1])
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data

    for i in range(len(data_list)):
        if random.random() > 0.5:
            data_list[i] = rotate_left(data_list[i])
        else:
            data_list[i] = rotate_right(data_list[i])
    return data_list

def random_color(data, rate=0.2):
    r1 = (random.random() - 0.5) * 2 * rate
    r2 = (random.random() - 0.5) * 2 * rate
    data = data * (1 + r2) + r1
    return data

class CropedRoughData(Dataset):
    def __init__(self, file_path, data_root, batch_size, istrain=True):
        super(CropedRoughData, self).__init__()
        self.data_root = data_root
        self.istrain = istrain
        self.batch_size = batch_size
        with open(file_path, 'r') as file:
            self.name_list = file.readlines()
        error_list = []
        self.name_list = [n.strip() for n in self.name_list if n.strip() not in error_list]
        print(len(self.name_list))
        self.z_size = 64
        self.xy_size = 256

    def __len__(self):
        if self.istrain:
            return len(self.name_list)
        else:
            return len(self.name_list)

    def process_imgmsk(self, data, mask):
        data = data.astype(float)
        data2 = data.copy()
        data2[data2 > 800] = 800
        data2[data2 < -100] = -100
        data2 = (data2 + 100) / 900
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048
        mask2 = (mask > 1).astype(int).astype(float)
        mask = (mask > 0).astype(int).astype(float)
        return data, data2, mask, mask2

    def augment(self, data, data2, mask, mask2):
        if random.random() > 0.5:
            result = random_flip([data, data2, mask, mask2])
            data, data2, mask, mask2 = result[0], result[1], result[2], result[3]
        if random.random() > 0.5:
            result = random_rotate([data, data2, mask, mask2])
            data, data2, mask, mask2 = result[0], result[1], result[2], result[3]
        # if random.random() > 0.5:
        #     data = random_color(data)
        #     data2 = random_color(data2)
        return data, data2, mask, mask2

    def do_interpolate_fromnpy(self, input, size):
        if len(input.shape) == 3:
            input = torch.from_numpy(input).unsqueeze(0).unsqueeze(1)
        else:
            input = torch.from_numpy(input).unsqueeze(0)
        input = input.float()
        # input = F.interpolate(input, size=(size[0]*4, size[1]*4, size[2]*4), mode='trilinear', align_corners=False)
        input = F.interpolate(input, size=size, mode='trilinear', align_corners=False)
        return input

    def __getitem__(self, item):
        index = item
        name = self.name_list[index]
        self.data = np.load(os.path.join(self.data_root, 'img', self.name_list[index]))
        self.mask = np.load(os.path.join(self.data_root, 'msk', self.name_list[index]))
        data, mask = self.data, self.mask
        data = data.astype(float)
        data, data2, mask, mask2 = self.process_imgmsk(data, mask)
        if self.istrain:
            data, data2, mask, mask2 = self.augment(data, data2, mask, mask2)
        _data = np.array([data, data2, mask, mask2])
        _data = self.do_interpolate_fromnpy(_data, size=(self.z_size, self.xy_size, self.xy_size))
        data, data2, mask, mask2 = _data[:, 0], _data[:, 1], _data[:, 2].int(), _data[:, 3].int()
        data = torch.cat([data, data2, mask.float()], dim=0)
        mask = mask2

        return data, mask, name
























