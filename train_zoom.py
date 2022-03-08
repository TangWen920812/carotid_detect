import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data import *
import torch.nn.functional as F
import Unet3d
import argparse

def Dice_loss(pred, label, isglobal=True, alpha=0.2, beta=0.8):
    N = pred.shape[0]
    pred = pred.view(N, -1)
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

def validation(model, valid_dataloader, optimizer):
    model.eval()
    dice_list, recall_list = [], []
    with torch.no_grad():
        for iter, pack in enumerate(valid_dataloader):
            data = pack[0].float()
            label = pack[1]
            pred = model(data)
            pred = F.sigmoid(pred)
            dice = Dice_coef(pred.detach().cpu(), label.detach().cpu(), threshold=0.5)
            dice_list.append(dice)
            recall = Recall_coef(pred.detach().cpu(), label.detach().cpu(), threshold=0.5)
            recall_list.append(recall)
            optimizer.zero_grad()
            print('iter: ', iter, '/', len(valid_dataloader))
            torch.cuda.empty_cache()

    return np.array(dice_list).mean(), np.array(recall_list).mean()

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    train_dataset = ZoomedNpyDataset(args.img_root, args.msk_root, args.txtfile_train, istrain=True)
    valid_dataset = ZoomedNpyDataset(args.img_root, args.msk_root, args.txtfile_valid, istrain=False)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) * args.max_epoches // 4

    model = Unet3d.Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    best_dice, best_recall = 0, 0
    train_dice = []
    for ep in range(args.max_epoches):
        model.train()
        print('epoch: ', ep)
        for iter, pack in enumerate(train_dataloader):
            data = pack[0].float()
            label = pack[1].cuda()
            pred = model(data)
            optimizer.zero_grad()
            loss = Dice_loss(F.sigmoid(pred), label)
            loss.backward()
            optimizer.step()
            if iter % 1 == 0:
                # optimizer.step()
                # optimizer.zero_grad()
                print(iter + ep * len(train_dataloader), '/', max_step, 'loss: ', loss.item())
            # torch.cuda.empty_cache()
            train_dice.append(Dice_coef(F.sigmoid(pred).detach().cpu(), label.detach().cpu(), threshold=0.5))
            del data, label
            torch.cuda.empty_cache()

        print('train dice: ', np.array(train_dice).mean())
        valid_dice, valid_recall = validation(model, valid_dataloader, optimizer)
        print('validation dice: ', valid_dice, 'validation recall: ', valid_recall)
        if valid_dice + valid_recall > best_dice + best_recall:
            best_dice = valid_dice
            best_recall = valid_recall
            print('new model on epoch: ', ep, 'best dice coef: ', best_dice,
                  'best recall coef: ', best_recall)
            torch.save(model.module.state_dict(),
                       os.path.join('./saved', 'Unet3d_%i_%.3f_%.3f.pth' % (ep, best_dice, best_recall)))
        print(' ')

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
    parser.add_argument("--npy_path", default="./useful_patch_13.npy", type=str)
    parser.add_argument("--session_name", default="Unet3d", type=str)
    parser.add_argument("--weights", default='./saved/Unet3d_170_0.373.pth', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    train(args)
















