import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_connections(pred, label, thres=0.1):
    with torch.no_grad():
        pred = (pred > thres).int()
        inter = pred * label
        union = pred + label
        max_pooling = torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        while True:
            new_inter = max_pooling(inter)
            new_inter[union < 0] = 0
            if (new_inter == inter).all():
                inter = new_inter
                break
            else:
                inter = new_inter
        return inter

def weighted_focal_loss(pred, label, weight_mask, gamma=2, beta=0.2):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    label = label.view(N, -1)
    weight_mask = weight_mask.view(N, -1)
    probs = torch.sigmoid(pred)
    pt = torch.where(label == 1, probs, 1 - probs)
    pb = torch.where(label == 1, 1 - probs**beta, 1 - (1 - probs)**beta)
    pb = torch.where(weight_mask == 1, pb, 1) * beta
    ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, label.float())
    loss = (torch.pow(1 - pt, gamma) * ce_loss * pb)
    # loss = ce_loss
    loss = loss.mean()
    return loss

def weighted_dice_loss(pred, label, weight_mask, alpha=0.5, beta=0.5):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    pred = torch.sigmoid(pred)
    label = label.view(N, -1)
    weight_mask = weight_mask.view(N, -1)
    weight_mask = torch.where(weight_mask == 1, 0.2, 1)

    inter = torch.sum(pred * label * weight_mask)
    union = torch.sum(pred * label * weight_mask + torch.relu(pred - label) * alpha * weight_mask+ \
                      torch.relu(label - pred) * beta * weight_mask)
    loss = (1 - (inter + 1e-3) / (union + 1e-3))

    return loss













