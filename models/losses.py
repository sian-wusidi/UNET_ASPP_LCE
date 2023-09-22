#!/usr/bin/env python3

from keras import backend as K
import numpy as np


def DiceLoss(y_true, y_pred, num_classes=4):
    y_pred = y_pred[:, :, :, 0:num_classes]
    intersection = K.sum(y_true * y_pred, axis=(0, 1, 2))  # (batchsize, 4)
    smooth = 1e-6
    dice = (2.*intersection + smooth) / (K.sum(y_true, axis=(0, 1, 2)
                                               ) + K.sum(y_pred, axis=(0, 1, 2)) + smooth)
    dice_mean = K.mean(dice, axis=-1)
    dice = 1 - dice_mean

    return dice


def DiceCoeff_Uncertain(y_true, y_pred, uncertainmask, smooth=1e-6):
    if uncertainmask != None:
        # interpolate the value using the predicted uncertainty
        y_pred = (K.ones_like(uncertainmask) - uncertainmask) * \
            y_true + uncertainmask * y_pred

    # want the dice coefficient should always be in 0 and 1
    intersection = K.sum(y_true * y_pred)
    dice = (2.*intersection) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    mask = K.cast(K.not_equal(K.sum(y_true) +
                  K.sum(y_pred) - intersection, 0), 'float32')

    return dice, mask


def DiceLoss_Uncertain(y_true, y_pred, num_classes=4, smooth=1e-6):
    dice = []
    if y_true != None:
        if num_classes != np.shape(y_true)[-1]:
            return None

    y_pred_labels = y_pred[:, :, :, 0:num_classes]
    y_pred_uncertain = y_pred[:, :, :,
                              num_classes:2*num_classes]  # pixel-level

    for index in range(num_classes):
        d, m = DiceCoeff_Uncertain(
            y_true[:, :, :, index], y_pred_labels[:, :, :, index], y_pred_uncertain[:, :, :, index])
        if m != 0:
            dice.append(d)

    dice_mutilabel = K.sum(dice)/(len(dice)+smooth)
    uncertain_reg = RegularizationLoss(y_true, y_pred)
    loss = 1 - dice_mutilabel + 3*uncertain_reg
    return loss


def RegularizationLoss(y_true, y_pred):
    num_classes = 4
    y_pred_uncertain = y_pred[:, :, :, num_classes:2*num_classes]
    reg_loss = (K.mean(-K.log(y_pred_uncertain[:, :, :, 0])) + K.mean(-K.log(y_pred_uncertain[:, :, :, 2])) + 0.1*K.mean(-K.log(
        y_pred_uncertain[:, :, :, 1])) + 0.15*K.mean(-K.log(y_pred_uncertain[:, :, :, 3])))/4  # give per ckass uncertainty regularization
    return reg_loss
