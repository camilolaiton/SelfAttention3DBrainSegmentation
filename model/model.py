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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
# import tensorflow_addons as tfa

tf.get_logger().setLevel('INFO')

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

def create_train_dataset(config:dict):
    # x_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    #     config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + 'orig',
    #     validation_split=0.1,
    #     subset='training',
    #     seed=12,
    #     image_size=config['IMAGE_SIZE'],
    #     batch_size=config['BATCH_SIZE'],
    #     follow_links=True,
    # )

    # y_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    #     config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + config['LABEL'],
    #     validation_split=0.1,
    #     subset='training',
    #     seed=12,
    #     image_size=config['IMAGE_SIZE'],
    #     batch_size=config['BATCH_SIZE'],
    #     follow_links=True,
    # )

    # return zip(x_train_dataset, y_train_dataset)

    color_mode = 'grayscale'

    data_gen_args = dict(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        # rotation_range=90,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        zoom_range=0.2
    )

    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + 'orig', 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode=color_mode,
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
    )

    msk_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + config['LABEL'], 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode=color_mode,
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
    )

    return zip(img_generator, msk_generator)

def create_validation_dataset(config:dict):
    # x_val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    #     config['DATASET_PATH'] + 'test/' + config['VIEW_TRAINIG'] + 'orig',
    #     validation_split=0.1,
    #     subset='validation',
    #     seed=12,
    #     image_size=config['IMAGE_SIZE'],
    #     batch_size=config['BATCH_SIZE'],
    #     follow_links=True,
    # )

    # y_val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    #     config['DATASET_PATH'] + 'test/' + config['VIEW_TRAINIG'] + config['LABEL'],
    #     validation_split=0.1,
    #     subset='validation',
    #     seed=12,
    #     image_size=config['IMAGE_SIZE'],
    #     batch_size=config['BATCH_SIZE'],
    #     follow_links=True,
    # )

    # return zip(x_val_dataset, y_val_dataset)

    color_mode = 'grayscale'

    data_gen_args = dict(
        rescale=1./255,
    )

    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'test/' + config['VIEW_TRAINIG'] + 'orig', 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode=color_mode,
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
    )

    msk_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'test/' + config['VIEW_TRAINIG'] + config['LABEL'], 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode=color_mode,
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
    )

    return zip(img_generator, msk_generator)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Slice', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='bone')
    plt.axis('off')
  plt.show()

def show_dataset(datagen, num=1):
    for i in range(0, num):
        image,mask = next(datagen)
        print(image.shape)

        if (num == 150):
            display([image[0], mask[0]])

def main():
    BATCH_SIZE = 64
    IMAGE_SIZE = (256, 256)
    DATASET_PATH = 'dataset/'
    VIEW_TRAINIG = 'axial/'
    LABEL = 'left-cerebellum-white-matter'

    config = {
        'DATASET_PATH': DATASET_PATH,
        'VIEW_TRAINIG': VIEW_TRAINIG,
        'LABEL': LABEL,
        'IMAGE_SIZE': IMAGE_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
    }
    
    train_gen = create_train_dataset(config=config)
    val_gen = create_validation_dataset(config=config)

    show_dataset(datagen=train_gen, num=150)

if __name__ == "__main__":
    main()