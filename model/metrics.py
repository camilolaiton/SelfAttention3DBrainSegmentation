import tensorflow as tf
from tensorflow.keras import backend as K

def IoU_coef(y_true, y_pred):
    T = K.flatten(y_true)
    P = K.flatten(y_pred)

    intersection = K.sum(T*P)
    IoU = (intersection + 1.0) / (K.sum(T) + K.sum(P) - intersection + 1.0)
    return IoU

def IoU_loss(y_true, y_pred):
    return -IoU_coef(y_true, y_pred)

smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)