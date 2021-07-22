import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1):
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3] + tf.reduce_sum(y_pred, axis=[1,2,3]))
    return tf.reduce_mean((2. * intersect + smooth) / (union + smooth), axis=0)