import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data import *
import torch.nn.functional as F
import Unet3d
import argparse
import SimpleITK as sitk
from scipy.ndimage import zoom

def Dice_coef(pred, label, threshold=0.1):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    label = label.view(N, -1)
    inter = torch.sum(2 * pred * label, dim=1)
    union = torch.sum(pred + label, dim=1)
    dice = (inter + 1) / (union + 1)
    return dice.numpy()

def Recall_coef(pred, label, threshold=0.1):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    label = label.view(N, -1)
    inter = torch.sum(pred * label, dim=1)
    union = torch.sum(label, dim=1)
    recall = (inter + 1) / (union + 1)
    return recall.numpy()

def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    valid_dataset = ZoomedNpyDataset(args.img_root, args.msk_root, args.txtfile_valid, istrain=False)
    # valid_dataset = NiiTestDataset(args.img_root, args.msk_root, args.txtfile_valid, istrain=False)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    model = Unet3d.Net()
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0)
    model = torch.nn.DataParallel(model).cuda()

    model.eval()
    for iter, pack in enumerate(valid_dataloader):
        data = pack[0].float()
        shape = pack[1].numpy()[0]
        # print(shape)
        # label = pack[1].cuda()
        pred = model(data)
        pred = pred.detach().cpu().numpy()
        # label = label.detach().cpu().numpy()
        name = valid_dataset.name_list[iter][:-4]
        pred = pred[0, 0]
        # label = label

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.astype('uint8')
        # pred = zoom(pred, zoom=(shape[0]/pred.shape[0], shape[1]/pred.shape[1], shape[2]/pred.shape[2]), order=0)
        # label = label.astype('uint8')
        pred = sitk.GetImageFromArray(pred)
        # label = sitk.GetImageFromArray(label)
        sitk.WriteImage(pred, './result/' + name + '.nii')
        # sitk.WriteImage(label, './result/' + name + '_label.nii')

        print(iter, '/', len(valid_dataset), name)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=401, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--img_root", default="./data/zoomed_img", type=str)
    parser.add_argument("--msk_root", default="./data/zoomed_msk", type=str)
    parser.add_argument("--txtfile_train", default="./train.txt", type=str)
    parser.add_argument("--txtfile_valid", default="./val.txt", type=str)
    parser.add_argument("--session_name", default="Unet3d", type=str)
    parser.add_argument("--weights", default='./saved/Unet3d_372_0.777_0.839.pth', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    test(args)


















