import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data2 import CropedRoughData
from model import ResUnet, BalancedDataParallel
import SimpleITK as sitk
from skimage import measure, morphology
from scipy import ndimage
import matplotlib.pyplot as plt

def Dice_loss(pred, label, isglobal=True, alpha=0.75, beta=0.25):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    pred = torch.sigmoid(pred)
    label = label.view(N, -1)
    if isglobal:
        inter = torch.sum(pred * label)
        union = torch.sum(pred * label + torch.relu(pred - label) * alpha + \
                          torch.relu(label - pred) * beta)
        loss = (1 - (inter + 1e-3) / (union + 1e-3))
    else:
        inter = torch.sum(pred * label, dim=1)
        union = torch.sum(pred * label + torch.relu(pred - label) * alpha + \
                          torch.relu(label - pred) * beta, dim=1)
        loss = (1 - (inter + 1e-3) / (union + 1e-3)).sum() / N
    return loss

def Dice_coef(pred, label, threshold=0.1):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    pred = torch.sigmoid(pred)
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    label = label.view(N, -1)
    inter = torch.sum(2 * pred * label)
    union = torch.sum(pred + label)
    dice = (inter + 1) / (union + 1)
    return dice

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

def Focal_loss(pred, label, gamma=2):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    label = label.view(N, -1)
    probs = torch.sigmoid(pred)
    pt = torch.where(label == 1, probs, 1 - probs)
    ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, label.float())
    loss = (torch.pow(1 - pt, gamma) * ce_loss)
    # loss = ce_loss
    loss = loss.mean()
    return loss

def train():
    max_epoches = 200
    batch_size = 4
    lr = 0.0001
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    file_path = '/media/tangwen/data/carotid_detection/croped2/train.txt'
    data_root = '/media/tangwen/data/carotid_detection/croped2'

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    model = ResUnet(channel=3, out_c=1, filters=[16, 16, 32, 64, 128], droprate=0.0)
    train_dataset = CropedRoughData(file_path=file_path, data_root=data_root, batch_size=batch_size, istrain=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                   pin_memory=True, drop_last=True)

    max_step = len(train_dataset) // batch_size * max_epoches
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # weights_dict = torch.load(os.path.join('./saved_model', 'resunet_69_8.1061.pth'))
    # model.load_state_dict(weights_dict, strict=False)
    # model = BalancedDataParallel(gpu0_bsz=1, dim=0, module=model).cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    for ep in range(0, max_epoches):
        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()
            mask = pack[1].float().cuda()

            pred = model(data)
            C = pred.shape[1]
            loss_list_dice, loss_list_focal = [], []
            for c in range(C):
                loss_list_dice.append(Dice_loss(pred[:, c], mask[:, c]))
                loss_list_focal.append(Focal_loss(pred[:, c], mask[:, c]))

            loss_weight = [1.0]
            loss_dice = sum([loss_list_dice[i] * loss_weight[i] for i in range(len(loss_list_dice))]) / len(loss_list_dice)
            loss_focal = sum([loss_list_focal[i] * loss_weight[i] for i in range(len(loss_list_focal))]) / len(loss_list_focal)
            loss = loss_dice + loss_focal
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset) // batch_size, '/', max_step,
                      'loss:', loss.item(), 'focal loss:', loss_list_focal[0].item(),
                      'dice loss:', loss_list_dice[0].item())
            torch.cuda.empty_cache()

        print('')
        dice_valid, recall_valid = validation(model)
        print('avg dice on validation set:', dice_valid)
        torch.save(model.module.state_dict(), os.path.join('./saved_model',
                                                           'resunet_' + str(ep) + '_%.4f_%.4f.pth' % (dice_valid, recall_valid)))

def lesion_measure(pred, label, thres=0.1):
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    pred = (pred > thres).astype(int)
    label_instance = measure.label(label, neighbors=8)
    gt = len(np.unique(label_instance)) - 1
    tp = len(np.unique(label_instance * pred)) - 1
    _pred = ndimage.binary_dilation(pred, np.ones((3,3,3)))
    pred_instance = measure.label(_pred, neighbors=8)
    all_p = len(np.unique(pred_instance)) - 1
    fp = max(0, all_p - tp)
    return tp, fp, gt

def validation(model):
    batch_size = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    file_path = '/media/tangwen/data/carotid_detection/croped2/test.txt'
    data_root = '/media/tangwen/data/carotid_detection/croped2'

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    valid_dataset = CropedRoughData(file_path=file_path, data_root=data_root, batch_size=batch_size, istrain=False)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, drop_last=False)
    dice_list = []
    tp, fp, gt = 0, 0, 0
    with torch.no_grad():
        for iter, pack in enumerate(valid_data_loader):
            data = pack[0].float().cuda()
            mask = pack[1].float().cuda()
            pred = model(data)
            N = pred.shape[0]

            mask = mask[:, 0]
            pred = pred[:, 0]
            for i in range(N):
                dice = Dice_coef(pred[i:i+1], mask[i:i+1])
                dice_list.append(dice.detach().cpu().numpy())
                _tp, _fp, _gt = lesion_measure(pred[i], mask[i], thres=0.1)
                tp += _tp
                fp += _fp
                gt += _gt
                print(iter, '/', len(valid_dataset)//batch_size, dice.detach().cpu().numpy().mean(),
                      'tp:', _tp, 'fp:', _fp, 'gt:', _gt)
            torch.cuda.empty_cache()

        valid_dice = np.array(dice_list).mean()
        recall = tp / gt
        percision = tp / (tp + fp)
        f1 = (recall * percision * 2) / (recall + percision + 1e-7)
        print('tp:', tp, 'fp:', fp, 'gt:', gt, 'recall:', recall, 'percision:', percision, 'f1:', f1)
    return valid_dice, recall




if __name__ == '__main__':
    train()
    









