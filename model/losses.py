import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import segmentation_models as sm
import tensorflow_addons as tfa
sm.set_framework('tf.keras')


def IoU_coef(y_true, y_pred):
    T = K.flatten(y_true)
    P = K.flatten(y_pred)

    intersection = K.sum(T*P)
    IoU = (intersection + 1.0) / (K.sum(T) + K.sum(P) - intersection + 1.0)
    return IoU

def IoU_loss(y_true, y_pred):
    return -IoU_coef(y_true, y_pred)

smooth=100

def dice_coef_3cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_3cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_3cat(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss

def dice_focal_loss(weights):
    dice_loss = sm.losses.DiceLoss(class_weights=np.array(weights)) 
    # focal_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
    focal_loss = sm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0)
    loss = dice_loss + (1*focal_loss)
    return loss

def dice_categorical(weights):
    dice_loss = sm.losses.DiceLoss(class_weights=np.array(weights)) 
    celoss = sm.losses.CategoricalCELoss(class_weights=np.array(weights))
    loss = dice_loss + (1*celoss)
    return loss

# https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.pys
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.5
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.3
    return K.pow((1-pt_1), gamma)

def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def generalized_dice_loss(weights):
    return sm.losses.DiceLoss(class_weights=np.array(weights))

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

def Combo_loss(targets, inputs, eps=1e-9):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
    
    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
    
    return combo

class FocalLoss(object):
    def __init__(self,  alpha=0.25, gamma=2):
        """
        :param alpha: A scalar tensor for focal loss alpha hyper-parameter
        :param gamma: A scalar tensor for focal loss gamma hyper-parameter
        """
        self.alpha = alpha
        self.gamma = gamma

    def get_loss(self, logits, labels, weights=None):
        """Compute focal loss for predictions.
                Multi-labels Focal loss formula:
                    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                         ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = labels.
            Args:
             logits: A float tensor of shape [batch_size,num_classes]
             labels: A float tensor of shape [batch_size,num_classes]
             weights: A float tensor of shape [batch_size, num_classes]
            Returns:
                loss: A (scalar) tensor representing the value of the loss function
        """
        sigmoid_p = tf.nn.sigmoid(logits)
        zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        # For poitive prediction, only need consider front part loss, back part is 0;
        pos_p_sub = tf.where(labels > zeros, labels - sigmoid_p, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # labels > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(labels > zeros, zeros, sigmoid_p)
        fl_loss = - self.alpha * (pos_p_sub ** self.gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                  - (1 - self.alpha) * (neg_p_sub ** self.gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

        if weights is None:
            loss = tf.reduce_sum(fl_loss*weights)/tf.maximum(tf.reduce_sum(weights), 1e-5)
        else:
            loss = tf.reduce_sum(fl_loss)

        return loss

# Boundary Loss
from scipy.ndimage import distance_transform_edt as distance


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


# # Scheduler
# ### The following scheduler was proposed by @marcinkaczor
# ### https://github.com/LIVIAETS/boundary-loss/issues/14#issuecomment-547048076

# class AlphaScheduler(Callback):
#     def __init__(self, alpha, update_fn):
#         self.alpha = alpha
#         self.update_fn = update_fn
#     def on_epoch_end(self, epoch, logs=None):
#         updated_alpha = self.update_fn(K.get_value(self.alpha))
#         K.set_value(self.alpha, updated_alpha)


# alpha = K.variable(1, dtype='float32')

# def gl_sl_wrapper(alpha):
#     def gl_sl(y_true, y_pred):
#         return alpha * generalized_dice_loss(
#             y_true, y_pred) + (1 - alpha) * surface_loss_keras(y_true, y_pred)
#     return gl_sl

# model.compile(loss=gl_sl_wrapper(alpha))

# def update_alpha(value):
#   return np.clip(value - 0.01, 0.01, 1)

# history = model.fit_generator(
#   ...,
#   callbacks=AlphaScheduler(alpha, update_alpha)
# )
