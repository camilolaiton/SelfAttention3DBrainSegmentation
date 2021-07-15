"""
    Made by:
        - Camilo Laiton
        Universidad Nacional de Colombia, Colombia
        2021-1
        GitHub: https://github.com/camilolaiton/

        This file belongs to the private repository "master_thesis" where
        I save all the files that are related to my thesis which is called
        "Método para la segmentación de imágenes de resonancia magnética 
        cerebrales usando una arquitectura de red neuronal basada en modelos
        de atención".
"""

import numpy as np
import os
import math
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
# import tensorflow_addons as tfa

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def build_model():
    # IMAGE_SIZE = 64
    # data_augmentation = keras.Sequential(
    #     [
    #         layers.experimental.preprocessing.Normalization(),
    #         layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    #         # layers.experimental.preprocessing.RandomFlip("horizontal"),
    #         layers.experimental.preprocessing.RandomRotation(factor=0.02),
    #         # layers.experimental.preprocessing.RandomZoom(
    #         #     height_factor=0.2, width_factor=0.2
    #         # ),
    #     ],
    #     name="data_augmentation",
    # )
    # Compute the mean and the variance of the training data for normalization.
    # data_augmentation.layers[0].adapt(x_train)

    pass

def get_train_test_dirs(prefix_path='data/', train_percentage=.8):
    roots = [
        'HLN-12',
        'Colin27',
        'MMRR-3T7T-2',
        'NKI-RS-22',
        'NKI-TRT-20',
        'MMRR-21',
        'OASIS-TRT-20',
        'Twins-2',
        'Afterthought'
    ]
    roots = [prefix_path + root for root in roots]
    final_roots = []

    structure = 'left-cerebellum-white-matter'
    for root in roots:
        for dir in os.listdir(root):
            final_roots.append((root + '/' + dir + '/slices/axial', root + '/' + dir + '/segSlices/' + structure + '/axial'))
    # images = glob(DATA_PATH)
    final_roots = shuffle(final_roots, random_state=12)

    data_size = len(final_roots)
    train_size = math.ceil(data_size*train_percentage)
    test_size = data_size - train_size

    train_dirs = final_roots[:train_size]
    test_dirs = final_roots[train_size:]

    return train_dirs, test_dirs

def main():
    train_dirs, test_dirs = get_train_test_dirs()
    print(test_dirs)


if __name__ == "__main__":
    main()